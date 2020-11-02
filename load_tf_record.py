import tensorflow as tf
from functools import partial
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

class TFRecordLoader():
    def __init__(self, batch_size, image_size):
        self.BATCH_SIZE = batch_size
        self.IMAGE_SIZE = image_size

    def decode_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, [self.IMAGE_SIZE, self.IMAGE_SIZE, 3])
        print("error")
        return image

    def read_tfrecord(self, example, labeled):
        tfrecord_format = (
            {
                "image/encoded": tf.io.FixedLenFeature([], tf.string),
                "image/class/label": tf.io.FixedLenFeature([], tf.int64),
            }
            if labeled
            else {"image/encoded": tf.io.FixedLenFeature([], tf.string),}
        )
        example = tf.io.parse_single_example(example, tfrecord_format)
        image = self.decode_image(example["image/encoded"])
        if labeled:
            label = tf.cast(example["image/class/label"], tf.int32)
            return image, label
        return image

    def load_dataset(self, filenames, labeled=True):
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed
        dataset = tf.data.TFRecordDataset(
            filenames
        )  # automatically interleaves reads from multiple files
        dataset = dataset.with_options(
            ignore_order
        )  # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(
            partial(self.read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
        )
        # returns a dataset of (image, label) pairs if labeled=True or just images if labeled=False
        return dataset

    def get_dataset(self, filenames, batch_size=16, labeled=True):
        dataset = self.load_dataset(filenames, labeled=labeled)
        #dataset = dataset.shuffle(2048)
        #dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        dataset = dataset.batch(batch_size)
        return dataset


    def show_batch(self, image_batch, label_batch):
        plt.figure(figsize=(10, 10))
        for n in range(25):
            ax = plt.subplot(5, 5, n + 1)
            plt.imshow(image_batch[n] / 255.0)
            if label_batch[n]:
                plt.title("MALIGNANT")
            else:
                plt.title("BENIGN")
            plt.axis("off")

