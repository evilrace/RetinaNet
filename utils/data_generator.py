from utils import data_preprocess
import tensorflow as tf
def get_generator(img_list, label_list):
    def generator():
        for img_path, label_path in zip(img_list, label_list):
            img = tf.image.decode_image(tf.io.read_file(img_path))
            label, _ = data_preprocess.decode_label(label_path)
            bboxes, clss_ids = data_preprocess.decode_train_label(label)
            yield img, bboxes, clss_ids
    return generator


def get_dataset(img_list, label_list):
    gen = get_generator(img_list, label_list)
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None, ), dtype=tf.float32),
        )
    )
    dataset = dataset.map(data_preprocess.prepare_data, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset