import os

from moviepy import VideoFileClip, clips_array


def join_videos_side_by_side(video_path1, video_path2, output_path="combined_video.mp4", **kwargs):
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
    fps = kwargs.get("fps", 30)
    final_video.write_videofile(output_path, codec="libx264", fps=fps)

    print(f"Combined video saved as {output_path}")


class RepresentationPostProcessor:
    """
    A class to handle post-processing of representation videos.

    Attributes:
        cfg (object): Configuration object containing settings.
        save_dir (str): Directory where the processed videos will be saved.
        exp_save_name (str): Experiment save name used for naming the output files.

    Methods:
        join_videos(original_video, representation_video, rep_type, dim_red_method):
            Joins the original and representation videos side by side and saves the output.

        cleanup(files_to_remove):
            Removes specified files if the clean_up_videos flag is set in the configuration.
    """

    def __init__(self, cfg, save_dir, exp_save_name):
        self.cfg = cfg
        self.save_dir = save_dir
        self.exp_save_name = exp_save_name

    def join_videos(self, original_video, representation_video, rep_type, dim_red_method):
        """
        Joins two videos side by side and saves the output.

        Parameters:
        original_video (str): Path to the original video file.
        representation_video (str): Path to the representation video file.
        rep_type (str): Type of representation (e.g., 'PCA', 't-SNE').
        dim_red_method (str): Dimensionality reduction method used.

        Returns:
        None
        """
        output_path = os.path.join(
            self.save_dir, f"{self.exp_save_name}_{rep_type}_{dim_red_method}.mp4"
        )
        join_videos_side_by_side(
            original_video,
            representation_video,
            output_path,
            fps=self.cfg.fps,
        )
        print(f"Joined video saved at {output_path}")

    def cleanup(self, files_to_remove):
        """
        Remove specified files if the clean_up_videos configuration is enabled.

        Args:
            files_to_remove (list): A list of file paths to be removed.

        Returns:
            None
        """
        if self.cfg.clean_up_videos:
            for file in files_to_remove:
                os.remove(file)
                print(f"Removed file: {file}")
