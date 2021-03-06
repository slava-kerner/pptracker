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


def print_class(cls):
    members = [attr for attr in dir(cls) if not callable(getattr(cls, attr)) and not attr.startswith("__")]
    for m in members:
        print(m, getattr(cls, m))


class TrackerBall(object):
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 0
        self.prev_gray = None

    def filter_by_color(self, img, blobs):
        color_tol = 10
        def is_ball(lab):
            # print(lab)
            return (127-color_tol < lab[1] < 127+color_tol) & (127-color_tol < lab[2] < 127+color_tol)

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # print('shape', lab.shape)
        return [blob for blob in blobs if is_ball(lab[int(blob.pt[1]), int(blob.pt[0])])]

    def run(self, video_path, video_out, out_rate=10):
        """
        
        :param video_path: 
        :param video_out: 
        :param out_rate: [Hz] 
        :return: 
        """
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
            self.frame_idx += 1

            if self.frame_idx < 180:  # here begins action
                continue

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
                        # print('min_move')
                        continue
                    if not good_flag:
                        # print('good_flag')
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
                # p = cv2.HoughCircles(frame_gray, cv2.HOUGH_GRADIENT, 2, 10, 5, 20)
                # p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)
                # print(p)
                # if p is not None:
                #     for x, y in p.astype(np.float32).reshape(-1, 2):
                #         # print(x, y)
                #         self.tracks.append([(x, y)])

            if self.prev_gray is not None and frame_gray is not None:
                diff = self.prev_gray.astype(np.float32) - frame_gray.astype(np.float32)

                # process diff:
                threshold = 5
                diff[abs(diff) < threshold] = 0  # keep objects with much change
                diff[frame_gray < np.max(frame_gray) * .75] = 0  # keep bright objects, like ball
                diff = diff.astype(np.uint8)
                diff[diff > 0] = 255
                diff = cv2.medianBlur(diff, 5)

                blob_detector_params = cv2.SimpleBlobDetector_Params()
                # print(blob_detector_params.minRepeatability)
                # print(blob_detector_params.thresholdStep)
                blob_detector_params.minRepeatability = 1
                blob_detector_params.minThreshold = 250
                blob_detector_params.maxThreshold = 255
                # blob_detector_params.thresholdStep = 10
                blob_detector_params.filterByArea = False
                blob_detector_params.maxArea = 100
                blob_detector_params.filterByConvexity = False
                blob_detector_params.filterByInertia = False
                blob_detector_params.minInertiaRatio = .1
                blob_detector_params.filterByColor = False
                # blob_detector_params.blobColor = 255
                blob_detector_params.filterByCircularity = True
                blob_detector_params.minCircularity = .5
                # blob_detector_params.minDistBetweenBlobs = 10
                # print_class(blob_detector_params)

                detector = cv2.SimpleBlobDetector_create(blob_detector_params)
                blobs = detector.detect(diff)
                # print('frame %d, detected %d blobs' % (self.frame_idx, len(blobs)))
                # for blob in blobs:
                #     print(blob.pt)
                # cv2.imwrite('output/frame%d.png' % int(self.frame_idx), frame)

                blobs = self.filter_by_color(vis, blobs)

                diff_with_keypoints = cv2.drawKeypoints(diff, blobs, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                if blobs is not None and blobs:
                    for blob in blobs:
                        # print(blob.pt)
                    # for x, y in p.astype(np.float32).reshape(-1, 2):
                #         print(x, y)
                        self.tracks.append([(blob.pt[0], blob.pt[1])])
                        # print_class(blob)
                    # print(blobs[0].pt[0])
                    blobs_txt = ', '.join(['%0f_%0f' % (blob.pt[1], blob.size) for blob in blobs])
                    cv2.putText(diff_with_keypoints, "%d tracks: %s" % (len(blobs), blobs_txt), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .75, 255)
                    time.sleep(1 / out_rate)

                # p = cv2.HoughCircles(diff, cv2.HOUGH_GRADIENT, 2, 10, 5, 20)
                # print(p)

                cv2.imshow('lk_track', diff_with_keypoints)
                if 180 < self.frame_idx < 210:
                    # cv2.imwrite('output/tracked_%d.png' % self.frame_idx, diff_with_keypoints)
                    lab = cv2.cvtColor(vis, cv2.COLOR_BGR2LAB)
                    cv2.imwrite('output/vis_%d.png' % self.frame_idx, frame)
                    cv2.imwrite('output/lab%d.png' % self.frame_idx, lab)
                    cv2.imwrite('output/diff%d.png' % self.frame_idx, diff)
                    cv2.imwrite('output/diff_with_keypoints%d.png' % self.frame_idx, diff_with_keypoints)
                    # cv2.imwrite('output/tracked_%d.png' % self.frame_idx, cv2.drawKeypoints(frame_gray, blobs, np.array([]), (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
            self.prev_gray = frame_gray
            # cv2.imshow('lk_track', vis)
            # print(vis.shape)
            cap_out.write(vis)

            # time.sleep(1 / fps)  # simulates real-time framerate
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

        cap.release()
        cap_out.release()

        cv2.destroyAllWindows()
