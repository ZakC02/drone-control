import os
import cv2


def decompose_video(video_path, output_folder, g):
    # Read the video file
    video = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize frame count
    frame_count = 0

    # Loop through the video frames
    while True:
        # Read the next frame
        ret, frame = video.read()

        # If the frame was not successfully read, then we have reached the end of the video
        if not ret:
            break

        # Save the frame as a JPEG image
        output_path = os.path.join(output_folder, f"frame_{g}_{frame_count:04d}.jpg")
        cv2.imwrite(output_path, frame)

        # Increment the frame count
        frame_count += 1

    # Release the video file
    video.release()

    print(f"Decomposition complete. {frame_count} frames extracted.")

def decompose_videos_in_folders(input_folder):
    # Get the list of subfolders (video folders)
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir()]

    # Iterate over the video folders
    for folder in subfolders:
        # Get the folder name (e.g., '1', '2', '3', etc.)
        folder_name = os.path.basename(folder)

        # Create the output folder for frames
        output_folder = os.path.join(input_folder, f"{folder_name}_frames")

        # Get the list of MP4 files in the video folder
        video_files = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith('.mp4')]

        # Iterate over the MP4 files and decompose each video
        g = 0
        for video_file in video_files:
            decompose_video(video_file, output_folder, g)
            g = g + 1

# Example usage
input_folder = "hands_version/videos"

decompose_videos_in_folders(input_folder)

