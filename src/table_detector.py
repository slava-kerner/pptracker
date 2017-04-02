import cv2
import numpy as np
import time
import copy
import functools
import operator
import math
from PIL import Image


class TableDetector(object):
    def __init__(self):
        pass

    def detect(self):
        """ Returns [UL, UR, BR, BL], each as [x, y]"""
        # currently hacking it manually, for track2:
        return np.array([[554, 250], [772, 250], [788, 382], [544, 382]], dtype=np.float32)

    def extract_lines(self, contours_path):
        img = cv2.imread(contours_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # edges = cv2.Canny(gray, 80, 120)
        lines = cv2.HoughLinesP(gray, 1, math.pi / 180, 20, 30, 10)
        print('detected %d lines' % len(lines))
        for line in lines:
            pt1 = (line[0][0], line[0][1])
            pt2 = (line[0][2], line[0][3])
            cv2.line(img, pt1, pt2, (0, 0, 255), 3)
        cv2.imwrite('output/lines.png', img)
        cv2.imshow('output/lines.png', img)

    def avg_contours(self, video_path):
        cap = cv2.VideoCapture(video_path)
        window_size = 3000
        threshold = .95
        contour = None
        counter = 0
        while (cap.isOpened()) & (counter < window_size):
            ret, frame = cap.read()
            if ret:
                if contour is None:
                    contour = self.detect_contours(frame).astype(np.float32)
                else:
                    contour += self.detect_contours(frame).astype(np.float32)
                counter += 1
                # ch = 0xFF & cv2.waitKey(1)
                # if ch == 27:
                #     break
        # contour = functools.reduce(operator.add, contours)
        contour = np.divide(contour, counter).astype(np.uint8)
        Image.fromarray(contour.astype(np.uint8)).save('output/contour_avg.png')
        contour[contour < threshold * 255] = 0
        contour[contour >= threshold * 255] = 255
        print('stats', np.mean(contour), np.max(contour))
        cv2.imshow('contour', contour)
        Image.fromarray(contour.astype(np.uint8)).save('output/contour_avg_binary.png')
        ch = 0xFF & cv2.waitKey(20)

    def detect_contours(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        cv2.imwrite('output/lab.png', lab)
        lower_white = np.array([110, 110, 100])
        upper_white = np.array([146, 146, 255])
        mask = cv2.inRange(lab, lower_white, upper_white)
        img = cv2.bitwise_and(img, img, mask=mask)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = gray #cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
        # cv2.imshow('thresh', gray)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        print('detected %d contours' % len(cnts))

        # rects = [self.detect_contour(c) for c in cnts]
        # if rects is not None and rects:
        #     print('detected rectangles', np.array(rects[rects is not None]).shape)

        img_cnts = copy.deepcopy(img)
        for c in cnts:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            # M = cv2.moments(c)
            # cX = int((M["m10"] / M["m00"]))
            # cY = int((M["m01"] / M["m00"]))
            cv2.drawContours(img_cnts, [c], -1, (0, 255, 0), 2)
        cv2.imshow("Image", img_cnts)

        return img_cnts


    def detect_contour(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        if len(approx) == 4:
            # (x, y, w, h) = cv2.boundingRect(approx)
            return c
        else:
            return None
