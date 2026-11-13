# extract_frames.py: dump an edited video to per-frame PNGs named like DAVIS (00000.png, 00001.png, ...)
import argparse
import os

import imageio.v2 as imageio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="edited .mp4 written by video/edit.py")
    parser.add_argument("--out_dir", required=True, help="output folder, e.g. outputs/video_frames/bear")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    reader = imageio.get_reader(args.video)
    n = 0
    for i, frame in enumerate(reader):
        imageio.imwrite(os.path.join(args.out_dir, f"{i:05d}.png"), frame)
        n += 1
    reader.close()
    print(f"[done] {n} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
