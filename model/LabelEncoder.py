import tensorflow as tf
from model.Anchor import AnchorBox
from utils.data_preprocess import *
def compute_iou(boxes1, boxes2):
    '''
    compute pairwise iou matrix for given two sets of boxes
    box types is xywh with shape [num_boxes, 4]
    '''
    boxes2 = convert_to_xywh(boxes2)
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lefts = tf.maximum(boxes1_corners[:,None,0], boxes2_corners[:,0])
    rights = tf.minimum(boxes1_corners[:,None,2], boxes2_corners[:,2])
    tops = tf.maximum(boxes1_corners[:,None,1], boxes2_corners[:,1])
    bottoms = tf.minimum(boxes1_corners[:,None,3], boxes2_corners[:,3])
    widths = tf.maximum(0.0, rights - lefts)
    heights = tf.maximum(0.0, bottoms - tops)
    intersection_areas = widths*heights

    boxes1_areas = boxes1[:,2] * boxes1[:,3]
    boxes2_areas = boxes2[:,2] * boxes2[:,3]

    union_areas = tf.maximum(boxes1_areas[:,None] + boxes2_areas - intersection_areas, 1e-8)
    iou = intersection_areas / union_areas
    return iou

class LabelEncoder:
    ''' encodes the raw labels of kitti data into targets for training'''

    def __init__(self) -> None:
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _match_anchor_boxes(self, anchor_boxes, gt_boxes, match_iou = 0.5, ignore_iou = 0.4):
    # '''
    # matches gt boxes to anchor boxes based on IOU
    # '''
        box_ioues = compute_iou(anchor_boxes, gt_boxes)
        matched_gt_idx = tf.argmax(box_ioues, axis=1)
        max_ioues = tf.reduce_max(box_ioues, axis=1)
        positive_mask = max_ioues > match_iou
        negative_mask = max_ioues < ignore_iou
        ignore_mask = ~ (positive_mask | negative_mask)

        return matched_gt_idx, tf.cast(positive_mask, dtype=tf.float32), tf.cast(ignore_mask, dtype=tf.float32)
        
        
    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):

        box_target = tf.concat(
            [
                (matched_gt_boxes[:,:2] - anchor_boxes[:,:2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis = -1,
        )

        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        '''
        '''
        anchor_boxes = self._anchor_box.get_anchors(image_shape[0], image_shape[1])
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        matched_gt_idx, positive_mask, ignore_mask = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )

        matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)

        cls_target = tf.where(
            tf.math.not_equal(positive_mask, True), -1.0, matched_gt_cls_ids
        )

        cls_target = tf.where(
            tf.math.equal(ignore_mask, True), -2.0, cls_target
        )

        cls_target = tf.where(
            tf.math.equal(cls_target, 8.0), -2.0, cls_target
        )

        cls_target = tf.expand_dims(cls_target, axis = -1)

        label = tf.concat([box_target, cls_target], axis=-1)

        return label


    def encode_batch(self, batch_images, batch_gt_boxes, batch_cls_ids):
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]
        images_shape = images_shape[1:]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=False)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, batch_gt_boxes[i], batch_cls_ids[i])
            labels = labels.write(i, label)
        batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        return batch_images, labels.stack()


