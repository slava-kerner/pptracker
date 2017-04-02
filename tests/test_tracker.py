import unittest

# from src.tracker import Tracker
from src.tracker_ball import TrackerBall
from src.table_detector import TableDetector

class TestTrackerBall(unittest.TestCase):
    @unittest.skip('')
    def test_show(self):
        TrackerBall().run('tests/data/track2.mp4', 'output/track2.avi')

    @unittest.skip('')
    def test_detect_table(self):
        det = TableDetector()
        # det.avg_contours('tests/data/track2.mp4')
        det.extract_lines('output/contour_avg_binary.png')