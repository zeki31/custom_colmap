# Merge all videos
from pathlib import Path

import moviepy

video_paths = sorted(Path("results/tracking_aliked_12fps10win_1").glob("*.mp4"))
video_clips = [moviepy.VideoFileClip(video_file) for video_file in video_paths]
final_video = moviepy.concatenate_videoclips(video_clips)
final_video.write_videofile(
    "results/tracking_aliked_12fps10win_1/merged_video.mp4", codec="libx264"
)
# self.logger.log({"video": wandb.Video(f"results/tracking_aliked_12fps10win_1/merged_video.mp4", fps=60, format="mp4")})
