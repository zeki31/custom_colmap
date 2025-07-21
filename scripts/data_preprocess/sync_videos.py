"""
Synchronize videos in a directory with the video that started recording most lately.

Original data structure:
- ordered
    - lane
        - running
            - 1_fixed.mp4
            - 2_dynA.mp4
            - 3_dynB.mp4
            - 4_dynC.mp4
            - 5_iphoneA.mp4
            - 6_iphoneB.mp4
        - running_cinematic
            - 1_fixed.mp4
            - ...

The expected output structure:
- synchronized
    - lane
        - running
            - 1_fixed.mp4
            - 2_dynA.mp4
            - 3_dynB.mp4
            - 4_dynC.mp4
            - 5_iphoneA.mp4
            - 6_iphoneB.mp4
        - running_cinematic
            - 1_fixed.mp4
            - ...

Example usage:
python scripts/data_preprocess/sync_videos.py -b ../datasets/sony_ai/ordered/lane/running
"""

import subprocess
from argparse import ArgumentParser
from pathlib import Path

import cv2
from tqdm import tqdm


class Synchronizer:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.output_dir = Path(str(base_dir).replace("ordered", "synchronized"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sync_videos(self) -> None:
        """Convert all videos in the base directory to frames."""
        video_paths = sorted(self.base_dir.glob("*.MP4")) + sorted(
            self.base_dir.glob("*.mov")
        )

        latest_start_t = 0
        for video_path in video_paths:
            start_second = self.get_start_timecode(video_path)
            if latest_start_t < start_second:
                latest_start_t = start_second

        for video_path in tqdm(video_paths, desc="Synchronizing videos"):
            start_second = self.get_start_timecode(video_path)
            fps = cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FPS)
            self.sync_one_video(video_path, latest_start_t - start_second, fps)

    def get_start_timecode(self, video_path: Path) -> float:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "d:0",
            "-show_entries",
            "stream_tags=timecode",
            "-of",
            "default=nw=1:nk=1",
            video_path,
        ]
        timecode = subprocess.run(cmd, capture_output=True, text=True)
        timecode_str = timecode.stdout.strip().replace(";", ":")
        hh, mm, ss, ms = map(int, timecode_str.split(":"))
        return hh * 3600 + mm * 60 + ss + ms / 1000

    def sync_one_video(self, video_path: Path, cut_duration: float, fps: float) -> None:
        """Convert a video to frames using FFMPEG."""
        output_video_path = self.output_dir / video_path.name

        command = [
            "ffmpeg",
            "-y",  # Overwrite output without asking
            "-i",
            str(video_path),
            "-ss",
            str(cut_duration),
            "-c:v",
            "copy",  # Copy video stream
            "-c:a",
            "aac",  # Re-encode audio to AAC
            "-r",
            str(fps),
            str(output_video_path),
        ]
        subprocess.run(command, check=True)


def main():
    parser = ArgumentParser(description="Convert videos to frames.")
    parser.add_argument("-b", "--base_dir", type=Path)
    args = parser.parse_args()

    synchronizer = Synchronizer(args.base_dir)
    synchronizer.sync_videos()


if __name__ == "__main__":
    main()
