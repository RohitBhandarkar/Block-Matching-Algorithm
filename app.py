import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

block_size = 32
motion_threshold = 15  # change threshold
total = 0
iframes = 25  # capture iframe
not_iframe = False  # capture iframe
count = 0
mode = "mse"

filename = "sec"  # filename
video_capture = cv2.VideoCapture(f"assets\\{filename}.mp4")
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
ret, anchor_frame = video_capture.read()
anchor_frame_gray = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY)


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


while True:
    ret2, target_frame = video_capture.read()
    total += 1
    if (total) % iframes != 0:
        not_iframe = True
    if not ret2 or not ret:
        break
    target_frame_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
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
    # if count>200:
    #     print(count,total)#,target_frame_gray,np.average(target_frame),sum(anchor_frame_gray-target_frame_gray))

    if not_iframe:
        cv2.imshow("Motion Detection", anchor_frame_gray)
        cv2.waitKey(1)
        not_iframe = False
    anchor_frame_gray = target_frame_gray
    count = 0

video_capture.release()
cv2.destroyAllWindows()
