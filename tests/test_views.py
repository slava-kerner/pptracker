import unittest

from src.table_detector import TableDetector
from src.views import Views


@unittest.skip('')
class TestViews(unittest.TestCase):
    def test_to_nadir(self):
        table_corners = TableDetector().detect()
        mode = 'nadir_ver'  # 'side_hor'
        table_ratio = .5
        video_in_path = 'tests/data/track2.mp4'
        video_out_path = 'output/tracks2_%s.avi' % mode

        view = Views(table_corners, mode, table_ratio)
        view.transform_video(video_in_path, video_out_path)
