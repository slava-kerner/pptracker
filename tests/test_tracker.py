import unittest

# from src.tracker import Tracker
from src.tracker_ball import TrackerBall

# @unittest.skip('')
class TestTrackerBall(unittest.TestCase):
    def test_show(self):
        TrackerBall().run('tests/data/track2.mp4', 'output/tracks2.avi')
