from utils import data_preprocess
import tensorflow as tf
def get_generator(img_list, label_list):
    def generator():
        for img_path, label_path in zip(img_list, label_list):
            img = tf.image.decode_image(tf.io.read_file(img_path))
            label, _ = data_preprocess.decode_label(label_path)
            label = data_preprocess.decode_train_label(label)
            yield img, label
    return generator


def get_dataset(img_list, label_list):
    gen = get_generator(img_list, label_list)
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(None, 5), dtype=tf.float32)
        )
    )
    # for sample in dataset:
    #     img, label = data_preprocess.prepare_data(sample[0], sample[1])
    #     print(img.shape, label.shape)
    dataset = dataset.map(data_preprocess.prepare_data)
    return dataset