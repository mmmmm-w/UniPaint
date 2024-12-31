import cv2
import numpy as np

def video_to_npy(video_path, npy_save_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read video frames and store them in a list
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:  # Break the loop if no more frames
            break
        
        # Convert frame from BGR to RGB (OpenCV reads as BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    # Release the video capture object
    cap.release()

    # Convert list of frames to a NumPy array with shape (f, h, w, c)
    video_array = np.array(frames)
    mask_array = np.ones_like(video_array)
    mask_array[:,114:397,:,:]=0
    mask_array = mask_array[:,:,:,:1]*255

    # Save the NumPy array to a .npy file
    np.save(npy_save_path, video_array)
    np.save("mask.npy", mask_array)
    print(f"Video saved to {npy_save_path} with shape {video_array.shape}")

# Example usage
video_to_npy("outpaint_videos/Compare/Compare_snow.mp4", "images.npy")