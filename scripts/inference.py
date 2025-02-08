# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
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
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
import cv2  # For resizing if needed

def apply_superresolution(subframe, ref_region, method):
    """
    Enhances the generated subframe if its resolution is lower than the reference region.
    
    Args:
        subframe (np.array): The generated lipsynced area.
        ref_region (np.array): The corresponding region from the original frame.
        method (str): 'gfpgan', 'codeformer', or 'both'.
    
    Returns:
        np.array: The enhanced subframe.
    """
    # Get the dimensions of the subframe and reference region
    sub_h, sub_w = subframe.shape[:2]
    ref_h, ref_w = ref_region.shape[:2]
    
    # Check if the subframe needs upscaling
    if sub_h < ref_h or sub_w < ref_w:
        # Determine upscale factor using the maximum ratio
        upscale_factor = max(ref_h / sub_h, ref_w / sub_w)
        if upscale_factor <= 1:
            return subframe  # No enhancement needed

        # Define helper functions for each superresolution model
        def run_gfpgan(image, factor):
            from gfpgan import GFPGANer  # Import GFPGANer
            model_path = "checkpoints/GFPGANv1.3.pth"  # Adjust path as necessary
            sr_model = GFPGANer(model_path=model_path, upscale=factor, arch='clean')
            enhanced, _ = sr_model.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
            return enhanced

        def run_codeformer(image, factor):
            from codeformer import CodeFormer  # Import CodeFormer
            model_path = "checkpoints/codeformer.pth"  # Adjust path as necessary
            sr_model = CodeFormer(model_path=model_path, upscale=factor)
            enhanced = sr_model.enhance(image)
            return enhanced

        # Convert method to lowercase for comparison
        method = method.lower()
        if method == "gfpgan":
            return run_gfpgan(subframe, upscale_factor)
        elif method == "codeformer":
            return run_codeformer(subframe, upscale_factor)
        elif method == "both":
            # If both methods are chosen, you can apply one after the other.
            # Here, we apply GFPGAN first, then CodeFormer.
            temp = run_gfpgan(subframe, upscale_factor)
            return run_codeformer(temp, upscale_factor)
        else:
            # If the method is unrecognized, use a fallback (bicubic interpolation)
            target_w = int(sub_w * upscale_factor)
            target_h = int(sub_h * upscale_factor)
            return cv2.resize(subframe, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    else:
        return subframe  # No upscaling needed



def main(config, args):
    # Check if the GPU supports float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    audio_encoder = Audio2Feature(model_path=whisper_model_path, device=device, num_frames=config.data.num_frames)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,  # load checkpoint
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    # set xformers
    if torch.cuda.is_available() and is_xformers_available():
       unet.enable_xformers_memory_efficient_attention()

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to(device)

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

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
        superres=args.superres 
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
    parser.add_argument("--superres", type=str, default="none",
                        help="Superresolution method to use: GFPGAN, CodeFormer, or both")
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    main(config, args)
