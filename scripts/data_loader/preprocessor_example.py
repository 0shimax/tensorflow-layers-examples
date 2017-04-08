import tensorflow as tf
import numpy as np


# def input_fn():
#     examples_op = tf.contrib.learn.read_batch_examples(
#         FILE_NAMES,
#         batch_size=100,
#         reader=tf.TFRecordReader(),
#         num_epochs=1,
#         parse_fn=lambda x: tf.decode_csv(x, [tf.constant([''], dtype=tf.string)] * len(HEADERS)))
#
#     examples_dict = {}
#     for i, header in enumerate(HEADERS):
#         examples_dict[header] = examples_op[:, i]
#
#     feature_cols = {k: tf.string_to_number(examples_dict[k], out_type=tf.float32)
#                     for k in CONTINUOUS_FEATURES}
#
#     feature_cols.update({k: dense_to_sparse(examples_dict[k])
#                          for k in CATEGORICAL_FEATURES})
#
#     label = tf.string_to_number(examples_dict[LABEL], out_type=tf.int32)
#
#     return feature_cols, label


class DataReader(tf.TFRecordReader):
    def __init__(self):
        super().__init__()

    def read(self, filename_queue):
        _, serialized_example = super().read(filename_queue)
        features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
              'image_raw': tf.FixedLenFeature([], tf.string),
              'label': tf.FixedLenFeature([], tf.int64),
          })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape([784])

        # OPTIONAL: Could reshape into a 28x28 image and apply distortions
        # here.  Since we are not applying any distortions in this
        # example, and the next step expects the image to be flattened
        # into a vector, we don't bother.

        # Convert from [0, 255] -> [-0.5, 0.5] floats.
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        # Convert label from a scalar uint8 tensor to an int32 scalar.
        label = tf.cast(features['label'], tf.int32)
        return self.augment(tf.reshape(image, [28,28,1])), label

    def augment(self, image):
        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(image, seed=1)

        # # Randomly crop a [height, width] section of the image.
        # distorted_image = tf.random_crop(distorted_image, [height, width, 3], seed=1)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        feature = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8, seed=1)
        # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63, seed=1)
        return feature



def input_to_model(file_list):
    features = {'raw_data': tf.FixedLenFeature([], tf.float32),
              'label': tf.FixedLenFeature([], tf.int64)}

    return tf.contrib.learn.read_batch_examples(
            file_list,
            batch_size=100,
            reader=DataReader,
            randomize_input=True,
            num_epochs=20,
            queue_capacity=10000,
            num_threads=1,
            read_batch_size=1,
            parse_fn=None,
            name=None,
        )


    # tensor_dict = tf.contrib.learn.read_batch_record_features(fileList,
    #                             batch_size=100,
    #                             features=features,
    #                             randomize_input=True,
    #                             reader_num_threads=4,
    #                             num_epochs=1,
    #                             name='input_pipeline')
    # tf.local_variables_initializer()
    # data = tensor_dict['raw_data']
    # labelTensor = tensorDict['label']
    # inputTensor = tf.reshape(data,[-1,10,10,1])
    # return inputTensor,labelTensor


def gen_image(images):
    for i in range(images.get_shape()[0]):
        yield augment(images[i])


def augment(image):
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(image, seed=1)

    # # Randomly crop a [height, width] section of the image.
    # distorted_image = tf.random_crop(distorted_image, [height, width, 3], seed=1)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    feature = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8, seed=1)
    # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63, seed=1)
    print(feature.dtype)
    return feature


def gen(data):
    for i in range(data.shape[0]):
        yield data[i]
