"""
Video to frames conversion script.
Original data structure:
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

The expected output structure:
- frames
    - lane
        - running
            - 1_fixed
                - images
                    - 00000.jpg
                    - 00001.jpg
                    - ...
            - 2_dynA
                - images
                    - 00000.jpg
                    - 00001.jpg
                    - ...

Example usage:
python scripts/data_preprocess/video2frames.py -b ../datasets/sony_ai/synchronized/lane/running
"""

import subprocess
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm


class Videos2Frames:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.output_dir = Path(str(base_dir).replace("synchronized", "frames"))

    def convert_videos(self) -> None:
        """Convert all videos in the base directory to frames."""
        video_paths = sorted(self.base_dir.glob("*.MP4"))
        video_paths = [p for p in video_paths if p.stem != "1_fixed_numbered"]

        for video_path in tqdm(video_paths, desc="Converting videos to frames"):
            self.convert_one_video(video_path, 59.94005994005994)

        # video_paths = sorted(self.base_dir.glob("*.mov"))
        # for video_path in tqdm(video_paths, desc="Converting videos to frames"):
        #     self.convert_one_video(video_path, 58.987787509392064)

    def convert_one_video(self, video_path: Path, fps: float) -> None:
        """Convert a video to frames using FFMPEG."""
        output_images_dir = self.output_dir / video_path.stem / "images"
        output_images_dir.mkdir(parents=True, exist_ok=True)

        command = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            str(output_images_dir / "%05d.jpg"),
        ]

        subprocess.run(command, check=True)


def main():
    parser = ArgumentParser(description="Convert videos to frames.")
    parser.add_argument("-b", "--base_dir", type=Path)
    args = parser.parse_args()

    converter = Videos2Frames(args.base_dir)
    converter.convert_videos()


if __name__ == "__main__":
    main()
