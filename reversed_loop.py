import cv2
import math
import subprocess


def process_video(input_file, output_file, desired_duration):
    # Open the input video file
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        raise IOError("Cannot open video file.")

    # Retrieve video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_duration = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate segments
    n_segments = math.ceil(desired_duration / clip_duration)
    if n_segments % 2 == 0:
        n_segments += 1

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("temp_output.mp4", fourcc, fps, (width, height))

    desired_total_frames = int(desired_duration * fps)
    written_frames = 0

    # Process segment by segment
    for segment in range(n_segments):
        if written_frames >= desired_total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start of video

        if segment % 2 == 0:  # Forward playback
            while True:
                ret, frame = cap.read()
                if not ret or written_frames >= desired_total_frames:
                    break
                out.write(frame)
                written_frames += 1
        else:  # Reverse playback
            frames_for_segment = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames_for_segment.append(frame)

            for frame in reversed(frames_for_segment):
                if written_frames >= desired_total_frames:
                    break
                out.write(frame)
                written_frames += 1

    cap.release()
    out.release()

    # Compress the output video
    compress_video("temp_output.mp4", output_file)

    # Clean up temporary file
    import os

    os.remove("temp_output.mp4")


def compress_video(input_path, output_path):
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        output_path,
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during ffmpeg execution: {e}")


if __name__ == "__main__":
    input_file = "/app/LatentSync/assets/stock_7.mp4"
    output_file = "demo_1.mp4"
    desired_duration = 30  # e.g., 10 seconds
    process_video(input_file, output_file, desired_duration)
