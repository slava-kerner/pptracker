import unittest

from src.tracker import Tracker


class TestTracker(unittest.TestCase):
    def test_show(self):
        Tracker().run('tests/data/track.mp4', 'output/tracks.avi')
