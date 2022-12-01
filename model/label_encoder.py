import tensorflow as tf
from .anchor import AnchorBox
from utils.data_preprocess import *
def compute_iou(boxes1, boxes2):
    '''
    compute pairwise iou matrix for given two sets of boxes
    box types are corner with shape [num_boxes, 4]
    '''
    boxes1 = convert_to_corners(boxes1)
    boxes2 = convert_to_corners(boxes2)


class LabelEncoder:
    ''' encodes the raw labels of kitti data into targets for training'''

    def __init__(self) -> None:
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtyope=tf.float32
        )

    def _match_anchor_boxes(self, anchor_boxes, gt_boxes, match_iou = 0.5, ignore_iou = 0.4):
    # '''
    # matches gt boxes to anchor boxes based on IOU
    # '''
        box_ious = compute_iou(anchor_boxes, gt_boxes)

