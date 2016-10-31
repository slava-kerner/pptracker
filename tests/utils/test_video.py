import unittest
import filecmp

from src.utils import video


class TestVideo(unittest.TestCase):
    def test_crop(self):
        in_path = 'tests/data/track.mp4'
        out_path = 'output/track.mp4'
        out_path_2 = 'output/track.mp4'
        video.crop(in_path, out_path, "00:00:00", "00:00:20")
        video.crop(out_path, out_path_2, "00:00:00", "00:00:20")
        self.assertTrue(filecmp.cmp(out_path, out_path_2))