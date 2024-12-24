from moviepy import VideoFileClip, clips_array


def join_videos_side_by_side(video_path1, video_path2, output_path="combined_video.mp4"):
    """
    Joins two videos side by side and saves the result as a new video.

    Args:
        video_path1 (str): Path to the first video file (MP4).
        video_path2 (str): Path to the second video file (MP4).
        output_path (str): Path to save the output video (default is "combined_video.mp4").

    Returns:
        None
    """
    # Load the two video clips
    video1 = VideoFileClip(video_path1)
    video2 = VideoFileClip(video_path2)

    # Resize both videos to the same height (optional, but recommended for alignment)
    height = min(video1.h, video2.h)  # Choose the smaller height of the two videos
    video1_resized = video1.resized(height=height)
    video2_resized = video2.resized(height=height)

    # Combine the videos side by side
    final_video = clips_array([[video1_resized, video2_resized]])

    # Write the result to a file
    final_video.write_videofile(output_path, codec="libx264", fps=24)

    print(f"Combined video saved as {output_path}")
