import imageio
import moviepy.editor as mpy
import numpy as np
import argparse

def main(args):
    input_video_path = args.input_video_path
    output_video_path = args.output_video_path
    start_frame = args.start_frame
    num_frames = args.num_frames 
    step = args.step
    
    # Read the video using imageio
    try:
        reader = imageio.get_reader(input_video_path, 'ffmpeg')
    except Exception as e:
        print(f"Error: Could not open video file {input_video_path}. Exception: {e}")
        return

    # Get video metadata
    fps = reader.get_meta_data()['fps']
    total_frames = reader.count_frames()
    print(f"Total frames in video: {total_frames}, FPS: {fps}")

    if start_frame >= total_frames:
        print(f"Error: Start frame {start_frame} exceeds the total number of frames in the video ({total_frames}).")
        return

    # Initialize a list to store the frames that we will extract and process
    processed_frames = []
    # Loop to process and extract exactly `num_frames`
    for frame_index in range(num_frames):
        # Calculate the exact frame to read
        current_frame_position = start_frame + frame_index * step

        if current_frame_position >= total_frames:
            print(f"Error: Reached the end of the video before processing the required frames.")
            break

        # Read the frame from the video
        try:
            frame = reader.get_data(current_frame_position)
        except IndexError:
            print(f"Error: Frame {current_frame_position} does not exist.")
            break

        # Extract the top square of the frame (assumes the video is rectangular)
        height, width, _ = frame.shape
        square_size = min(height, width)
        square_frame = frame[-square_size:, 0:square_size]

        # Resize the square frame to 512x512
        resized_frame = mpy.ImageClip(square_frame).resize(newsize=(512, 512)).get_frame(0)
        
        # Append the processed frame to the list
        processed_frames.append(resized_frame)

    # If any frames were processed, write them to the output video
    if processed_frames:
        # Use moviepy to write the processed frames into a video
        video_clip = mpy.ImageSequenceClip(processed_frames, fps=8)
        video_clip.write_videofile(output_video_path, codec="libx264")
        print(f"New video created successfully with {len(processed_frames)} frames.")
    else:
        print("No frames were processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a video by extracting square frames and saving as a new video.")
    parser.add_argument("--input_video_path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--output_video_path", type=str, required=True, help="Path to the output video file.")
    parser.add_argument("--start_frame", type=int, default=250, help="Frame number to start from.")
    parser.add_argument("--num_frames", type=int, default=16, help="Number of frames to process.")
    parser.add_argument("--step", type=int, default=2, help="Step size for frame sampling.")
    
    args = parser.parse_args()

    main(args)