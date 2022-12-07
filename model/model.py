import tensorflow as tf
from tensorflow import keras
import numpy as np 
from model.Anchor import AnchorBox
from utils import data_preprocess

def get_backbone():
    backbone = keras.applications.ResNet50(
        include_top = False, input_shape = [None, None, 3]
    )

    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output for layer_name in ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
    ]

    return keras.Model(
        inputs=[backbone.inputs], outputs = [c3_output, c4_output, c5_output]
    )


class FeaturePyramid(keras.layers.Layer):

    def __init__(self, backbone=None, **kwargs):
        super().__init__(name='FeaturePyramid', **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_1x1_layers = [keras.layers.Conv2D(256, 1, 1, 'same') for i in range(3)]
        self.conv_3x3_layers = [keras.layers.Conv2D(256, 3, 1, 'same') for i in range(3)]
        self.conv_3x3_layers2 = [keras.layers.Conv2D(256, 3, 2, 'same') for i in range(2)]
        self.upsample_2x = keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        outputs = self.backbone(images, training=training)

        p_outputs = [conv(output) for conv, output in zip(self.conv_1x1_layers, outputs)]
        p_outputs[1] = p_outputs[1] + self.upsample_2x(p_outputs[2])
        p_outputs[0] = p_outputs[0] + self.upsample_2x(p_outputs[1])
        p_outputs = [conv(output) for conv, output in zip(self.conv_3x3_layers, p_outputs)]
        p2_output = self.conv_3x3_layers2[0](outputs[-1])
        p3_output = self.conv_3x3_layers2[1](tf.nn.relu(p2_output))

        return (*p_outputs, p2_output, p3_output)

def build_head(output_filters, bias_init):

    head = keras.Sequential([keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01, seed=42)
    for _ in range(4):
        head.add(
            keras.layers.Conv2D(256, 3, padding='same', kernel_initializer = kernel_init, activation='relu')
        )
    head.add(keras.layers.Conv2D(
        output_filters,
        3,
        1,
        padding='same',
        kernel_initializer = kernel_init,
        bias_initializer = bias_init
    ))

    return head

class RetinaNet(keras.Model):
    def __init__(self, num_classes, num_anchors = 9, backbone=None, **kwargs):
        super().__init__(**kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        pior_probability = tf.constant_initializer(-np.log((1-0.01) / 0.01))
        self.cls_head = build_head(self.num_anchors * self.num_classes, pior_probability)
        self.box_head = build_head(self.num_anchors * 4, 'zeros')

    def call(self, image, training= False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []

        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(tf.reshape(self.cls_head(feature), [N, -1, self.num_classes]))
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis = -1)


class DecodePredictions(tf.keras.layers.Layer):
    ''' decodes predictions of the RetinaNet model'''
    def __init__(self, num_classes, confidence_threshold=0.05, nms_iou_threshold = 0.5, max_detections_per_class = 100, max_detections = 100, 
    box_variance = [0.1, 0.1, 0.2, 0.2], **kwargs):

        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            box_variance, dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        box_predictions = box_predictions * self._box_variance
        box_target = tf.concat(
            [
                (box_predictions[:,:,:2] * anchor_boxes[:, :, 2:] + anchor_boxes[:,:,:2]),
                tf.math.exp(box_predictions[:,:,2:]) * anchor_boxes[:, :, 2:],
            ],
            axis = -1,
        )
        box_target = data_preprocess.convert_to_corners(box_target)
        return box_target

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)[1:]
        anchor_boxes = self._anchor_box.get_anchors(image_shape[0], image_shape[1])
        box_predictions = predictions[:,:,:4]
        cls_predictions = tf.nn.sigmoid(predictions[:,:,4:])
        anchor_boxes = tf.expand_dims(anchor_boxes, axis=0)
        boxes = self._decode_box_predictions(anchor_boxes, box_predictions)
        
        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes = False,
        )