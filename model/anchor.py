import tensorflow as tf

class AnchorBox:

    '''Generate anchor boxes.
        this class generate anchor boxes for given strides, scales, aspect_ratios and size.
        Size is the length of one side of a rectangle. Anchor box creates a variable box holding area (size*size) with different aspect_ratio.
        the anchor box format is [x, y, width, height].
    '''
    
    def __init__(self, aspect_ratios:list[float]=[0.5,1.0,2.0], scales:list[float]=[0, 1/3, 2/3],  sizes:list[float] = [32.0, 64.0, 128.0, 256.0, 512.0]) -> None:
        self._aspect_ratios = aspect_ratios
        self._scales = [2 ** x for x in scales]
        self._num_anchors = len(self._aspect_ratios) * len(self._scales)
        self._areas = [size**2 for size in sizes]
        self._strides = [2 ** i for i in range(3,8)]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):

        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self._aspect_ratios:
                anchor_height = tf.math.sqrt(area/ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self._scales:
                    anchor_dims.append(scale * dims)

            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all


    def _get_anchors(self, feature_height, feature_width, level):

        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5

        centers = tf.meshgrid(rx, ry)
        centers = tf.stack(centers, axis =-1)
        centers = centers * self._strides[level-3]
        
        centers = tf.expand_dims(centers, axis =-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])

        centers_shape = tf.shape(centers)
        dims = tf.tile(
            self._anchor_dims[level-3], [centers_shape[0], centers_shape[1], 1, 1]
        )
        anchors = tf.concat([centers, dims], axis =-1)

        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        '''
        generarte anchor boxes for all feature maps
        '''
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i
            )
            for i in range(3,8)
        ]

        return tf.concat(anchors, axis=0)