# LatentSync with Superresolution Enhancement

This repository is a modified version of the original **latentsync** project. In this version, we have integrated a superresolution enhancement step to improve the quality of the generated (lipsynced) region using GFPGAN and CodeFormer. We have also adapted the code to automatically use GPU when available and fall back to CPU otherwise. This single commit includes improvements across several files. Below is a detailed explanation of every change and the overall process.

---

## Overview

### Key Improvements:
- **Superresolution Integration:**  
  A new command-line parameter `--superres` (with options: `GFPGAN`, `CodeFormer`, `both`, or `none`) has been added. When enabled, the generated lipsynced region is upscaled if its resolution is lower than that of the original video frame.

- **Adaptive Device Handling:**  
  The code now dynamically selects the device—using GPU if available and automatically falling back to CPU if not. Changes include:
  - Setting the device dynamically in `inference.py`
  - Passing the device as a string (e.g., `"cpu"`) to libraries (such as `face_alignment`) that require a string.

- **Robust Face Detection:**  
  In the `ImageProcessor` (located in `latentsync/utils/image_processor.py`):
  - For the `"fix_mask"` mode, if mediapipe fails to detect a face, a warning is printed and a default landmark set is returned to prevent a crash.
  - The input image is properly converted from a tensor (with shape `(C, H, W)`) to a NumPy array in `(H, W, C)` order and then from BGR to RGB before face detection.

- **Affine Transformation Stability:**  
  In `affine_transform.py`, the function `transformation_from_points` has been updated to:
  - Convert input landmark arrays to `float64` and center them.
  - Check if the standard deviation of the landmark points is too small (e.g., less than `1e-6`) and, if so, return an identity transformation.
  - Wrap the SVD computation in a try/except block to catch non-convergence issues and fall back to an identity transformation.
  - Optionally smooth the translation bias to stabilize the transformation.

- **FFmpeg Python Bindings:**  
  The incorrect `ffmpeg` package was removed and replaced with `ffmpeg-python` (v0.2.0), ensuring that the expected API (e.g., `ffmpeg.input`) is available for audio loading.

---

## Detailed Changes by File

### 1. `inference.sh`
- **What Changed:**  
  - The `--superres` parameter has been added to the command-line arguments.
- **Why:**  
  - This enables the selection of a superresolution method to enhance the generated lipsynced region.
- **How It Works:**  
  - The script passes all parameters (including `--superres`) to `scripts/inference.py`.

### 2. `scripts/inference.py`
- **Device Handling:**  
  - A new variable is defined as follows:
    ```python
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ```
    This ensures that the pipeline uses GPU if available, or falls back to CPU automatically.
- **Audio Encoder:**  
  - The device passed to `Audio2Feature` is set dynamically using the `device` variable.
- **Passing Superresolution Parameter:**  
  - The call to the pipeline now includes `superres=args.superres`.
- **Why:**  
  - This guarantees proper device selection and passes the chosen superresolution method to the pipeline.

### 3. `latentsync/pipelines/lipsync_pipeline.py`
- **ImageProcessor Initialization:**  
  - The instantiation of `ImageProcessor` now passes the device as a string:
    ```python
    self.image_processor = ImageProcessor(height, mask=mask, device="cuda" if torch.cuda.is_available() else "cpu")
    ```
  - **Why:**  
    - Certain libraries (e.g., `face_alignment`) expect the device as a string and will error if provided with a `torch.device` object.

### 4. `latentsync/utils/image_processor.py`
- **Initialization (`__init__`):**
  - **For `"fix_mask"` mode:**  
    - **Before (CPU branch):**
      ```python
      # self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
      self.face_mesh = None
      self.fa = None
      ```
    - **After (CPU branch):**
      ```python
      self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
      self.fa = None
      ```
    - **Why:**  
      - This ensures that mediapipe’s FaceMesh is properly initialized on the CPU to detect facial landmarks.
- **Face Detection (`detect_facial_landmarks`):**
  - **Change:**  
    - Instead of raising an error when no face is detected, the method now prints a warning and returns a default set of landmark coordinates (e.g., the center of the image repeated as needed).
  - **Why:**  
    - This prevents the pipeline from crashing on frames where a face is not detected.
- **Affine Transformation (`affine_transform`):**
  - **Change:**  
    - The input image is converted from tensor (C, H, W) to a NumPy array in (H, W, C) using `rearrange`, calling `.numpy()` only when necessary.
    - The image is then converted from BGR to RGB using `cv2.cvtColor` so that mediapipe receives the expected format.
  - **Why:**  
    - Mediapipe requires an RGB image in HWC format for proper landmark detection.

### 5. `latentsync/utils/affine_transform.py`
- **`transformation_from_points` Function:**
  - **Changes:**
    - Both landmark arrays are converted to `float64` and centered.
    - If the standard deviation of either set is too low (below `1e-6`), a warning is printed and an identity transformation is returned.
    - The SVD computation is wrapped in a try/except block. If SVD fails to converge, a warning is printed and an identity transformation is returned.
    - Optionally, smoothing is applied to the translation bias.
  - **Why:**  
    - When the landmark data is degenerate (for example, when default landmarks are used because no face was detected), SVD might not converge. This fallback mechanism prevents a crash and allows the pipeline to continue processing.

### 6. FFmpeg Python Bindings
- **Changes:**
  - The incorrect `ffmpeg` package was uninstalled, and `ffmpeg-python` (version 0.2.0) was installed.
  - This ensures that the expected API (e.g., `ffmpeg.input`, `ffmpeg.Error`) is available for loading audio.
- **Why:**  
  - The project expects the ffmpeg-python API for correct audio processing.

---

## How the Process Works

1. **Environment Setup:**
   - Clone the repository.
   - Create and activate a virtual environment.
   - Install dependencies from `requirements.txt`.
   - Ensure FFmpeg is installed (via conda or manually) and available in your system PATH.

2. **Running Inference:**
   - The `inference.sh` script passes all parameters (including the `--superres` flag) to `scripts/inference.py`.
   - `inference.py` dynamically selects the device (GPU if available, otherwise CPU) and passes it to the model initializations (Audio2Feature, U-Net, etc.).
   - The pipeline in `lipsync_pipeline.py` processes each video frame:
     - The `ImageProcessor` converts the frame to the proper format (RGB, HWC) and uses mediapipe’s FaceMesh to detect facial landmarks.
     - If no face is detected, a warning is printed and default landmarks are used.
     - The affine transformation is computed using `transformation_from_points`. Fallback mechanisms (identity transformation) are employed if landmark data is degenerate.
     - If the generated region is lower resolution than the corresponding original region, superresolution is applied using the selected method (GFPGAN, CodeFormer, or both).
   - Finally, an output video (e.g., `video_out.mp4`) with enhanced lipsync quality is produced.

3. **Output:**
   - The final video is saved as specified by the `--video_out_path` parameter.
   - Warnings may be printed during processing if default landmarks or fallback transformations are used, but the pipeline completes successfully.

---



## Requirements

- **Python 3.7 or higher**
- **Dependencies:**  
  See `requirements.txt` for all dependencies. Key packages include:
  - `torch`
  - `diffusers`
  - `ffmpeg-python`
  - `mediapipe`
  - `face_alignment`
  - `numpy`
  - `opencv-python`
  - `einops`
  - `torchvision`
- **FFmpeg:**  
  FFmpeg must be installed and available in your system PATH. On Windows, you can install it via conda:
  ```bash
  conda install -c conda-forge ffmpeg
  ```
  or download a build from ffmpeg.org and add its bin folder to your PATH.

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Bhagawat8/LatentSync_VideoDubber.git
   cd LatentSync_VideoDubber
   ```

2. **Create and Activate a Virtual Environment:**
   - On Unix/Linux/Mac or Git Bash on Windows:
     ```bash
     python -m venv env
     source env/Scripts/activate
     ```
   - On Windows Command Prompt:
     ```batch
     python -m venv env
     env\Scripts\activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   > **Tip:** If any dependency is missing, manually install packages such as `ffmpeg-python`, `mediapipe`, and `face_alignment`.

---

## Running Inference

The provided `inference.sh` script runs the lipsync inference pipeline with superresolution enhancement.

**Usage Example:**
```bash
./inference.sh --unet_config_path "configs/unet/second_stage.yaml" \
               --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
               --inference_steps 20 \
               --guidance_scale 1.5 \
               --video_path "assets/demo1_video.mp4" \
               --audio_path "assets/demo1_audio.wav" \
               --video_out_path "video_out.mp4" \
               --superres "both"
```

**Parameters:**
- `--unet_config_path`: Path to the U-Net configuration YAML.
- `--inference_ckpt_path`: Path to the U-Net checkpoint.
- `--inference_steps`: Number of inference steps.
- `--guidance_scale`: Guidance scale for inference.
- `--video_path`: Input video file.
- `--audio_path`: Input audio file (if the video does not have embedded audio, extract it using FFmpeg).
- `--video_out_path`: Output video file path.
- `--superres`: Superresolution method (options: `GFPGAN`, `CodeFormer`, `both`, or `none`).
