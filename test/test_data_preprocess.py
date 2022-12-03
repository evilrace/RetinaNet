import unittest
import utils.data_preprocess as data_preprocess
import numpy as np 
import tensorflow as tf
class TestPrepareData(unittest.TestCase):

    def test_unmatched_file_name(self):
        with self.assertRaises(NameError):
            data_preprocess.prepare_data(r"F:\dataset\kitti\data_object_image_2\training\image_2\000003.png", r"F:\dataset\kitti\data_object_image_2\training\label_2\000004.txt")

        
    def test_matched_file_name(self):
        try:
            data_preprocess.prepare_data(r"F:\dataset\kitti\data_object_image_2\training\image_2\000003.png", r"F:\dataset\kitti\data_object_image_2\training\label_2\000003.txt")
        except Exception as e:
            self.fail(str(e))

class TestBoxConvert(unittest.TestCase):
    def test_convert_xywh(self):
        box = tf.convert_to_tensor(
            [[100, 100, 200, 200],
            [-100, 100, 0, 200],
            [100, -100, 200, 0],
            [-100, -100, 0, 0]]
        )
        box_xywh = data_preprocess.convert_to_xywh(box)
        box_xywh = tf.cast(box_xywh, tf.int32)
        
        box_gt = tf.convert_to_tensor(
            [[150, 150, 100, 100],
            [-50, 150, 100, 100],
            [150, -50, 100, 100],
            [-50, -50, 100, 100]]
        )
        result = all(tf.reshape(box_gt, [-1]) == tf.reshape(box_xywh, [-1]))
        self.assertEqual(result, True)