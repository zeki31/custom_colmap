# Merge all videos
from pathlib import Path

import moviepy

video_paths = sorted(Path(f"results/tracking_aliked_{1}").glob("*.mp4"))
video_clips = [moviepy.VideoFileClip(video_file) for video_file in video_paths]
final_video = moviepy.concatenate_videoclips(video_clips)
final_video.write_videofile(
    f"results/tracking_aliked_{1}/merged_video.mp4", codec="libx264"
)
# self.logger.log({"video": wandb.Video(f"results/tracking_aliked_{1}/merged_video.mp4", fps=60, format="mp4")})
