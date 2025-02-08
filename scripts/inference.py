# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import img2tensor, tensor2img
from gfpgan import GFPGANer
from codeformer import CodeFormerRestorer  # Requires codeformer package

class SuperResolution:
    """Handles super-resolution using GFPGAN or CodeFormer"""

    def __init__(self, method):
        self.method = method
        self.model = None
        if self.method == "GFPGAN":
            self.model = GFPGANer(
                model_path="checkpoints/GFPGANv1.3.pth",
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
        elif self.method == "CodeFormer":
            self.model = CodeFormerRestorer(
                codeformer_path="checkpoints/codeformer.pth", upscale=1
            )

    def enhance(self, region_pil):
        """Apply super-resolution to a given region"""
        region_cv2 = cv2.cvtColor(np.array(region_pil), cv2.COLOR_RGB2BGR)
        if self.method == "GFPGAN":
            _, _, output = self.model.enhance(region_cv2, paste_back=True)
        elif self.method == "CodeFormer":
            output = self.model.enhance(region_cv2, fidelity_ratio=0.8)

        return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    def process_frame(self, frame, mouth_subframe, orig_bbox):
        """Replace the mouth region with an enhanced version"""
        x1, y1, x2, y2 = orig_bbox
        orig_width, orig_height = x2 - x1, y2 - y1
        gen_width, gen_height = mouth_subframe.size
        scale_w, scale_h = orig_width / gen_width, orig_height / gen_height
        scale_factor = max(scale_w, scale_h)

        if scale_factor > 1 and self.model:
            resized = mouth_subframe.resize(
                (int(gen_width * scale_factor), int(gen_height * scale_factor)),
                Image.BICUBIC,
            )
            enhanced = self.enhance(resized)
            final_subframe = enhanced.resize((orig_width, orig_height), Image.LANCZOS)
        else:
            final_subframe = mouth_subframe.resize((orig_width, orig_height), Image.LANCZOS)

        frame.paste(final_subframe, (x1, y1))
        return frame


def main(config, args):
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path, device="cuda", num_frames=config.data.num_frames
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model), args.inference_ckpt_path, device="cpu"
    )

    unet = unet.to(dtype=dtype)

    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    # Initialize super-resolution if needed
    superres = SuperResolution(args.superres) if args.superres else None

    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        superres=superres,  # Pass super-resolution object to pipeline
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument(
        "--superres",
        type=str,
        choices=["GFPGAN", "CodeFormer"],
        default=None,
        help="Super-resolution method for mouth region",
    )

    args = parser.parse_args()
    config = OmegaConf.load(args.unet_config_path)
    main(config, args)
