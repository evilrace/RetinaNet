import tensorflow as tf
import pathlib

def random_flip_horizontal(image, boxes):
    '''
    flip image and boxes horizontally.
    image : (height, width, channels)
    boxes : (num_boxes, 4), (left, top, right, bottom)
    '''
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:,2], boxes[:, 1], 1 - boxes[:,0], boxes[:3]], axis=-1
        )
        return image, boxes


def resize_and_pad_image(
    image, min_side = 800, max_side = 1333, jitter = [640, 1024], stride = 128):
    image_shape = tf.cast(tf.shape(image), tf.float32)[:2]

    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)

    ratio = min_side / tf.reduce_min(image_shape) 
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape) 
    
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))

    padded_image_shape = tf.cast( tf.math.ceil(image_shape/stride) * stride, dtype=tf.int32)
    image = tf.image.pad_to_bounding_box(image, 0, 0, padded_image_shape[0], padded_image_shape[1])

    return image, image_shape, ratio

def decode_label(label_file_path):
    '''
    decode kitti dataset label
    
    arguments:
        label_file_path: kitti dataset label file path
    returns:
        label_dict_list: list of label dictionary dtype as below
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                            'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                            'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                            truncated refers to the object leaving image boundaries
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                            0 = fully visible, 1 = partly occluded
                            2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                            contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    '''
    label_lists = []
    raw_label_lists = open(label_file_path,'r').readlines()
    label_dict_list = []
    for raw_label in raw_label_lists:
        label = raw_label.split()
        class_type = label[0]
        truncated = label[1]
        occluded = label[2]
        alpha = label[3]
        bbox = label[4:8]
        dimensions = label[8:11]
        location = label[11:14]
        rotation_y = label[14]
        label_dict = {'class_type': class_type, 'truncated':truncated, 'occluded':occluded, 'alpha':alpha, 'bbox':bbox, 'dimensions':dimensions, 'location': location,
        'rotation_y':rotation_y}
        label_dict_list.append(label_dict)


    return label_dict_list


def prepare_data(img_path, label_path):
    img_file_name = pathlib.PurePath(img_path).stem
    label_file_name = pathlib.PurePath(label_path).stem
    if img_file_name != label_file_name:
        raise Exception('The label file and image file name is unmatched.')

    img = tf.image.decode_image(tf.io.read_file(img_path))
    bbox = None


# prepare_data(r"F:\dataset\kitti\data_object_image_2\training\image_2\000003.png", r"F:\dataset\kitti\training\label_2\000003.txt")
label = decode_label(r"F:\dataset\kitti\training\label_2\000003.txt")
print(label)