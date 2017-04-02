import cv2
import numpy as np
import time


class Views(object):
    def __init__(self, table_corners, mode, table_ratio):
        """
        :param table_corners:
        :param mode: among ['nadir', 'side', ..]
        :param table_ratio: amount of the view taken by table, [0..1]. 1 means we only see table.
        """
        self.table_corners = table_corners
        self.mode = mode
        self.table_ratio = table_ratio

    def is_horizontal(self, mode):
        return mode.endswith('hor')

    def _get_corners(self, img_width, img_height):
        # print(img_width, img_height)
        if self.mode == 'nadir_ver':
            x_min = (1. - self.table_ratio) * img_width / 2.
            y_min = (1. - self.table_ratio) * img_height / 2.
            x_max = img_width - x_min
            y_max = img_height - y_min
            return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
        elif self.mode == 'nadir_hor':
            # img_height, img_width = img_width, img_height
            x_min = (1. - self.table_ratio) * img_width / 2.
            y_min = (1. - self.table_ratio) * img_height / 2.
            x_max = img_width - x_min
            y_max = img_height - y_min
            # return np.array([[y_min, x_max], [y_max, x_max], [y_max, x_min], [y_min, x_min]], dtype=np.float32)
            return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
        elif self.mode == 'side_hor':
            x_ratio = 1
            y_ratio = 2
            x_min = (1. - self.table_ratio * x_ratio) * img_width / 2.
            y_min = (1. - self.table_ratio * y_ratio) * img_height / 2.
            x_max = img_width - x_min
            y_max = img_height - y_min
            return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
        else:
            raise ValueError('mode %s not implemented' % self.mode)

    def transform_frame(self, img):
        """

        :param img: frame to be moved to view
        :return: reprojected view image
        """
        dst_points = self._get_corners(img.shape[0], img.shape[1])
        # print('dst_points', dst_points)
        # h = cv2.findHomography(self.table_corners.reshape(-1, 1, 2), dst_points.reshape(-1, 1, 2), cv2.RANSAC, 5.0)
        h = cv2.getPerspectiveTransform(self.table_corners, dst_points)
        # print(h)
        transformed = cv2.warpPerspective(img, h, (img.shape[0], img.shape[1]), flags=cv2.INTER_CUBIC)
        if self.is_horizontal(self.mode):
            transformed = cv2.transpose(transformed)
            # transformed = cv2.flip(transformed, 1)
            # return cv2.warpPerspective(img, h, (img.shape[1], img.shape[0]))
        # else:
        #     return cv2.warpPerspective(img, h, (img.shape[0], img.shape[1]))
        return transformed

    def transform_video(self, video_path, video_out_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if self.is_horizontal(self.mode):
            cap_out = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height), isColor=True)
        else:
            cap_out = cv2.VideoWriter(video_out_path, fourcc, fps, (height, width), isColor=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                return

            frame_out = self.transform_frame(frame)
            cv2.imshow('original', frame)
            cv2.imshow(self.mode, frame_out)
            cap_out.write(frame_out)
            # time.sleep(3/fps)
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

