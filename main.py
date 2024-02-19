import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(
    description="Set parameters for motion detection algorithm."
)

# Add arguments
parser.add_argument(
    "-b", "--block_size", type=int, help="Block size for motion detection"
)
parser.add_argument(
    "-t", "--motion_threshold", type=float, help="Threshold for motion detection"
)
parser.add_argument("-i", "--iframe_intervals", type=int, help="Interval for I-frames")
parser.add_argument("-m", "--mode", type=str, help="Mode of operation")
parser.add_argument("-w", "--write", type=bool, help="Write output to file")
parser.add_argument("-f", "--filename", type=str, help="Output filename")

args = parser.parse_args()

block_size = args.block_size if args.block_size else 32  # change block size
motion_threshold = (
    args.motion_threshold if args.motion_threshold else 15
)  # change threshold
iframes = (
    args.iframe_intervals if args.iframe_intervals else 25
)  # change iframe interval
_write = args.write if args.write else True  # create output video
mode = (
    args.mode if args.mode else "mse"
)  # measuring: Mean Square Error (mse) or Mean Absolute Difference (mad)
filename = (
    args.filename if args.filename else "sec"
)  # filename of the video in the assets folder

total = 0
not_iframe = False
count = 0
frames = []
vid_len = 1

video_capture = cv2.VideoCapture(f"assets\\{filename}.mp4")
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
ret, anchor_frame = video_capture.read()
anchor_frame_gray = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY)
anchor_frame_gray = cv2.GaussianBlur(anchor_frame_gray, (19, 19), 0)


def mse(block1, block2):
    _block1, _block2 = block1, block2
    _mse = mean_squared_error(_block1, _block2)
    return _mse


def mad(block1, block2):
    t1 = np.average(block1)
    t2 = np.average(block2)
    _mad = abs(t1 - t2)
    return _mad


def write(frames, path="out.avi", fps=30):
    H, W = frames[0].shape
    # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fourcc = -1
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H), True)
    for frame in frames:
        writer.write(frame)
    print("[INFO] Video Export Completed")


for f in tqdm(range((int)(length * vid_len))):
    ret2, target_frame = video_capture.read()
    total += 1
    if (total) % iframes != 0:
        not_iframe = True
    if not ret2 or not ret:
        break
    target_frame_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
    target_frame_gray = cv2.GaussianBlur(target_frame_gray, (19, 19), 0)
    anchor_buffer = anchor_frame_gray
    target_buffer = target_frame_gray
    for y in range(0, anchor_frame_gray.shape[0] - block_size + 1, block_size):
        for x in range(0, anchor_frame_gray.shape[1] - block_size + 1, block_size):
            anchor_block = anchor_frame_gray[y : y + block_size, x : x + block_size]
            target_block = target_frame_gray[y : y + block_size, x : x + block_size]
            op = (
                mad(anchor_block, target_block)
                if mode == "mad"
                else mse(anchor_block, target_block)
            )
            if op > motion_threshold:
                cv2.rectangle(
                    anchor_frame_gray,
                    (x, y),
                    (x + block_size, y + block_size),
                    (255, 255, 255),
                    2,
                )
                count += 1
    if not_iframe:
        frames.append(anchor_frame_gray)
        cv2.imshow("Motion Detection", anchor_frame_gray)
        cv2.waitKey(1)
        not_iframe = False
    anchor_frame_gray = target_frame_gray
    count = 0

if _write:
    write(frames, f'OUTPUT/{filename+"-bma"}.mp4', 15)
video_capture.release()
cv2.destroyAllWindows()
