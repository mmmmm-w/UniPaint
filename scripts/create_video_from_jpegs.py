import os
from moviepy.editor import ImageSequenceClip
import numpy as np
from moviepy.video.fx.all import crop, resize

# Root folders for input images and output videos
image_root_folder = "data/DAVIS/JPEGImages/Full-Resolution/"  # Root folder where the images are stored
output_video_root = "outpaint_videos/"  # Root folder for the output videos

# Ensure the output folder exists
os.makedirs(output_video_root, exist_ok=True)

# Function to sample 16 frames with adjustable intervals
def sample_frames(image_files, target_frames=16, default_interval=2):
    total_frames = len(image_files)

    if total_frames < target_frames:
        raise ValueError("Not enough frames in the folder!")

    # Try sampling with the default interval first
    interval = default_interval

    # Calculate the maximum number of frames we can sample with the current interval
    while (target_frames - 1) * interval + 1 > total_frames:
        interval -= 1
        if interval == 0:
            raise ValueError("Could not find a valid interval for sampling frames.")

    # Calculate the starting index to center the frames
    max_coverage = (target_frames - 1) * interval + 1
    start_index = (total_frames - max_coverage) // 2

    # Sample frames at the computed interval
    indices = [start_index + i * interval for i in range(target_frames)]
    return [image_files[i] for i in indices]

# Iterate over all subfolders in the root image folder
for subfolder in sorted(os.listdir(image_root_folder)):
    subfolder_path = os.path.join(image_root_folder, subfolder)
    
    # Ensure it's a directory (skip files)
    if os.path.isdir(subfolder_path):
        # Get the list of image files in the subfolder
        image_files = sorted([os.path.join(subfolder_path, img) for img in os.listdir(subfolder_path) if img.endswith(".jpg")])

        # Ensure there are enough images
        if len(image_files) < 16:
            print(f"Skipping {subfolder} as it contains less than 16 images.")
            continue

        try:
            # Sample 16 frames with an adjustable interval
            selected_frames = sample_frames(image_files)

            # Create a video clip from the selected frames
            clip = ImageSequenceClip(selected_frames, fps=8)  # Adjust fps if necessary

            # Apply center cut and resize to 512x512 for each frame
            clip = clip.fx(crop, x_center=clip.w / 2, y_center=clip.h / 2, width=min(clip.w, clip.h), height=min(clip.w, clip.h))
            clip = clip.fx(resize, newsize=(512, 512))

            # Create the output video file path using the subfolder name
            output_video_path = os.path.join(output_video_root, f"DA_{subfolder}.mp4")

            # Write the video file
            clip.write_videofile(output_video_path, codec='libx264', fps=8)  # Adjust fps of the output video here

            print(f"Created video for {subfolder} at {output_video_path}.")
        except ValueError as e:
            print(f"Error processing {subfolder}: {e}")