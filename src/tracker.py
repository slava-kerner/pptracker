#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
from src.utils.common import draw_str
import cv2
import time
import os


lk_params = dict(winSize=(5, 5), maxLevel=0, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 20))

feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=3, blockSize=2)

config = {'min_move': 1}


class Tracker(object):
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0
        self.prev_gray = None

    def run(self, video_path, video_out):
        assert os.path.exists(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        cap_out = cv2.VideoWriter(video_out, fourcc, fps, (width, height), isColor=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                return

            orig_frame = frame.copy()
            orig_frame_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            for b in range(frame.shape[2]):
                frame[:, :, b] = clahe.apply(frame[:, :, b])
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.array([tr[-1] for tr in self.tracks]).astype(np.float32).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), (x_old, y_old), good_flag in \
                        zip(self.tracks, p1.reshape(-1, 2), p0.reshape(-1, 2), good):
                    dist = np.sqrt((x - x_old) ** 2 + (y - y_old) ** 2)
                    if dist < config['min_move']:  # ignore objects that moved less than that many pixels
                        continue
                    # print(dist)
                    if not good_flag:
                        continue
                    rad = 4
                    xmin = int(max(0, x - rad))
                    ymin = int(max(0, y - rad))
                    xmax = int(min(width, x + rad))
                    ymax = int(min(height, y + rad))
                    if xmin == xmax or ymin == ymax \
                            or np.min(img0[ymin:ymax, xmin:xmax]) == 0 \
                            or np.min(orig_frame_gray[ymin:ymax, xmin:xmax]) == 0 \
                            or np.min(img1[ymin:ymax, xmin:xmax]) == 0:  # black, meaning boundaries -> ignore
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in p.astype(np.float32).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)
            # print(vis.shape)
            cap_out.write(vis)

            time.sleep(1 / fps)  # simulates real-time framerate
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

        cap.release()
        cap_out.release()

        cv2.destroyAllWindows()
