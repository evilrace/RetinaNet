import unittest
import data_preprocess
class TestPrepareData(unittest.TestCase):

    def test_unmatched_file_name(self):
        with self.assertRaises(Exception):
            data_preprocess.prepare_data(r"F:\dataset\kitti\data_object_image_2\training\image_2\000004.png", r"F:\dataset\kitti\training\label_2\000003.txt")
        try:
            data_preprocess.prepare_data(r"F:\dataset\kitti\data_object_image_2\training\image_2\000003.png", r"F:\dataset\kitti\training\label_2\000003.txt")
        except Exception:
            self.fail('assert is raised')
        