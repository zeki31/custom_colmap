"""
Remove the first N frames and the last M frames from each video,
and re-index the remaining frames.

Example usage:
python scripts/data_preprocess/cut_invalid_frames.py -b ../datasets/sony_ai/synchronized/lane/running -f 415 -l 1680
"""

import subprocess
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm


class FrameCutter:
    def __init__(self, base_dir: Path, first_n: int, last_n: int):
        self.base_dir = base_dir
        self.output_dir = Path(str(base_dir).replace("synchronized", "valid_charuco"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # frame number to seconds
        self.first_s = first_n / 59.94005994005994
        self.duration_s = last_n / 59.94005994005994 - self.first_s

    def cut_frames(self) -> None:
        """Cut the first N and last M frames from each video."""
        video_paths = sorted(self.base_dir.glob("*.MP4"))
        video_paths = [
            p
            for p in video_paths
            if p.stem != "1_fixed_numbered"
            and p.stem != "5_iphoneA"
            and p.stem != "6_iphoneB"
        ]

        for video_path in tqdm(video_paths, desc="Cutting frames from videos"):
            self.cut_one_video(video_path)

    def cut_one_video(self, video_path: Path) -> None:
        """Cut frames from a single video."""
        output_video_path = self.output_dir / video_path.name

        command = [
            "ffmpeg",
            "-ss",
            str(self.first_s),
            "-i",
            str(video_path),
            "-t",
            str(self.duration_s),
            "-c:v",
            "copy",  # Copy video stream
            "-c:a",
            "aac",  # Re-encode audio to AAC
            str(output_video_path),
        ]

        subprocess.run(command)


def main():
    parser = ArgumentParser(description="Cut invalid frames from videos.")
    parser.add_argument(
        "-b",
        "--base_dir",
        type=Path,
        required=True,
        help="Base directory containing videos.",
    )
    parser.add_argument(
        "-f",
        "--first_n",
        type=int,
        default=0,
        help="Number of frames to cut from the start.",
    )
    parser.add_argument(
        "-l",
        "--last_n",
        type=int,
        default=0,
        help="Number of frames to cut from the end.",
    )

    args = parser.parse_args()

    cutter = FrameCutter(args.base_dir, args.first_n, args.last_n)
    cutter.cut_frames()


if __name__ == "__main__":
    main()
