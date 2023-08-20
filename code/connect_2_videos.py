import cv2
import os
import argparse

def glue_videos_together(video1_path: str, video2_path: str, output_path: str, delete_originals: bool = False):
    # Create a new video writer for the combined video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = (256, 256)
    fps = 5.0
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Open each video in sequence and write its frames to the combined videopyt
    for video_path in [video1_path, video2_path]:
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()

    # Optionally, delete the original videos
    if delete_originals:
        os.remove(video1_path)
        os.remove(video2_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Glue two videos together.")
    parser.add_argument("video1_path", type=str, help="Path to the first video.")
    parser.add_argument("video2_path", type=str, help="Path to the second video.")
    parser.add_argument("output_path", type=str, help="Path to the output video.")
    parser.add_argument("--delete", action="store_true", help="Delete the original videos after gluing.")

    args = parser.parse_args()

    glue_videos_together(args.video1_path, args.video2_path, args.output_path, args.delete)
