from argparse import ArgumentParser

import wandb

"""
python scripts/upload_mp4_wandb.py -v results/colmap.webm -i ert4kq3f
"""

parser = ArgumentParser()
parser.add_argument(
    "-i", "--id", type=str, required=True, help="Run ID to upload the video to"
)
parser.add_argument(
    "-v", "--video", type=str, required=True, help="Path to the video file"
)
args = parser.parse_args()

with wandb.init(
    project="custom_colmap",
    entity="zeki31-global-page",
    id=args.id,
    resume="allow",  # reconnect to that exact run
) as run:
    # Log your MP4
    run.log(
        {
            "COLMAP GUI": wandb.Video(
                args.video,
                format="webm",
            )
        }
    )
