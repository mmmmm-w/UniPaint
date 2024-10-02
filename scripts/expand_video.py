import imageio
import numpy as np
from PIL import Image
import os

def process_video(video_path, interval):
    # Read the video
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    
    # Get video properties from the first frame
    first_frame = reader.get_data(0)
    frame_height, frame_width, _ = first_frame.shape
    
    # Check if the video frame dimensions are valid
    if frame_width == 0 or frame_height == 0:
        print(f"Error: Video frame dimensions are invalid (width={frame_width}, height={frame_height}).")
        return
    
    # Determine the new square size before scaling
    square_size = max(frame_width, frame_height)
    
    # Prepare the mask matrix for each frame
    mask = np.ones((square_size, square_size), dtype=np.uint8)
    y_offset = (square_size - frame_height) // 2
    x_offset = (square_size - frame_width) // 2
    
    # Set the original area in the mask to 0
    mask[y_offset:y_offset + frame_height, x_offset:x_offset + frame_width] = 0
    
    # Resize the mask matrix to 512x512
    mask_image = Image.fromarray(mask)
    resized_mask = mask_image.resize((512, 512), resample=Image.NEAREST)
    
    # Save the resized mask as a .npz file
    mask_filename = os.path.splitext(video_path)[0] + ".npz"
    np.savez(mask_filename, mask=np.array(resized_mask))
    packed_mask = np.packbits(np.array(resized_mask))
    # Save the packed data to a compressed npz file
    np.savez_compressed(mask_filename, mask=packed_mask)
    print(f"Mask saved as {mask_filename}")
    
    # Prepare the output video writer (512x512 resolution)
    output_path = video_path.replace(".mp4", "_512x512.mp4")
    writer = imageio.get_writer(output_path, fps=8)
    
    # Sample frames based on the interval
    total_frames = reader.count_frames()
    frame_indices = range(0, total_frames, interval)[:16]  # Get the first 16 frames with the interval
    
    # Iterate through the selected frame indices
    for frame_count in frame_indices:
        frame = reader.get_data(frame_count)
        
        # Convert the frame to PIL image
        frame_image = Image.fromarray(frame)
        
        # Create a black square background
        square_frame = Image.new('RGB', (square_size, square_size), (0, 0, 0))
        
        # Paste the original frame into the center of the square
        square_frame.paste(frame_image, (x_offset, y_offset))
        
        # Resize the square frame to 512x512
        resized_frame = square_frame.resize((512, 512), resample=Image.LANCZOS)
        
        # Convert back to numpy array and write to the output video
        writer.append_data(np.array(resized_frame))
    
    # Close the video writer and reader
    writer.close()
    reader.close()
    
    print(f"Processed video saved as {output_path}")

# Usage example with a given interval
video_path = "./outpaint_videos/CI/CI_Robot.mp4"
interval = 1  # For example, sample every 30th frame
process_video(video_path, interval)