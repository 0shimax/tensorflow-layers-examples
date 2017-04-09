import os, sys
import tensorflow as tf
import numpy as np
from contextlib import ExitStack


class DataReader(tf.TFRecordReader):
    def __init__(self):
        super().__init__()

    def read(self, filename_queue):
        key, serialized_example = super().read(filename_queue)
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

        capacity = 20 + 3 * 100
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size= 100,
            capacity=capacity,
            min_after_dequeue=20
        )
        return [images, labels]
        # return "key", ([self.augment(tf.reshape(image, [28,28,1])), label]),


    def augment(self, image):
        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(image, seed=1)
        # image = tf.image.random_flip_up_down(image)


        # # Randomly crop a [height, width] section of the image.
        # distorted_image = tf.random_crop(distorted_image, [height, width, 3], seed=1)
        # image = tf.image.resize_image_with_crop_or_pad(image, framesize, framesize)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        feature = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8, seed=1)
        # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63, seed=1)
        return feature


def input_to_model(file_list):
    features = {'raw_data': tf.FixedLenFeature([], tf.float32),
              'label': tf.FixedLenFeature([], tf.int64)}

    images, labels = tf.contrib.learn.read_batch_examples(
            file_list,
            batch_size=100,
            reader=DataReader,
            randomize_input=True,
            num_epochs=20,
            queue_capacity=10000,
            num_threads=1,
            read_batch_size=1,
            parse_fn=None,
        )

    return images, labels


def mnist_batch_input_fn(dataset, batch_size=100, seed=555, num_epochs=1):
    # If seed is defined, this will shuffle data into batches

    np_labels = np.asarray( dataset[1], dtype=np.int32)

    # Instead, build a Tensor dict from parts
    all_images = tf.constant( dataset[0], shape=dataset[0].shape, verify_shape=True )
    all_labels = tf.constant( np_labels,      shape=np_labels.shape, verify_shape=True )

    # And create a 'feeder' to batch up the data appropriately...

    image, label = tf.train.slice_input_producer( [all_images, all_labels],
                                                num_epochs=num_epochs,
                                                shuffle=(seed is not None), seed=seed,
                                              )

    image = tf.reshape(image, [28, 28, 1])
    # label = tf.one_hot(label, depth=10)

    dataset_dict = dict( images=image, labels=label ) # This becomes pluralized into batches by .batch()

    batch_dict = tf.train.batch( dataset_dict, batch_size,
                              num_threads=1, capacity=batch_size*2,
                              enqueue_many=False, shapes=None, dynamic_pad=False,
                              allow_smaller_final_batch=True,
                              shared_name=None, name=None)

    return batch_dict.pop('images'), batch_dict.pop('labels')


def file_input_fn(image_paths, op_labels, n_class, distortion=True,\
        batch_params={'size': 100, 'min_after_dequeue': 256, 'num_preprocess_threads':1}):
    image_fqueue = tf.train.string_input_producer(image_paths)

    reader = tf.WholeFileReader()
    key, value = reader.read(image_fqueue)
    img = tf.image.decode_jpeg(value, channels=3)

    img = tf.reshape(img, [118, 210, 3])
    img.set_shape((118, 210, 3))
    # img = tf.cast(img, tf.float32)

    one_hot_labels = tf.one_hot(op_labels, depth=n_class)

    # features = tf.parse_single_example(value, features={
    #     'label': tf.FixedLenFeature([], tf.int64),
    #     'image_raw': tf.FixedLenFeature([], tf.string),
    # })
    # label = tf.cast(features['label'], tf.int32)
    # image = tf.decode_raw(features['image_raw'], tf.int32)

    if distortion:
        img = augment(img)

    # Generate batch
    imgs, labels = tf.train.shuffle_batch(
        [img, one_hot_labels],
        batch_size=batch_params['size'],
        num_threads=batch_params['num_preprocess_threads'],
        capacity=batch_params['min_after_dequeue'] + 3 * batch_params['size'],
        min_after_dequeue=batch_params['min_after_dequeue'] )

    return imgs, labels


def image_path_getter(image_pointer, data_root_path='./data'):
    image_pointer_path = os.path.join(data_root_path, image_pointer)
    with open(image_pointer_path, 'r') as f:
        return [file_name.rstrip() for file_name in f]


def label_op_getter(label_file_name, data_root_path='./data'):
    label_file_path = os.path.join(data_root_path, label_file_name)
    with open(label_file_path, 'r') as f:
        return tf.constant( [int(label.rstrip()) for label in f] )


def augment(image):
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(image, seed=1)
    # image = tf.image.random_flip_up_down(image)


    # # Randomly crop a [height, width] section of the image.
    # distorted_image = tf.random_crop(distorted_image, [height, width, 3], seed=1)
    # image = tf.image.resize_image_with_crop_or_pad(image, framesize, framesize)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    feature = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8, seed=1)
    # distorted_image = tf.image.random_brightness(distorted_image, max_delta=63, seed=1)
    return feature


def minibatch_loader(image_list_file, label_file, data_root, \
                        n_class=10, batch_size=100, num_epochs=200):
    # Reads pfathes of images together with their labels
    image_list, label_list = \
        read_labeled_image_list(image_list_file, label_file, data_root)

    images = tf.convert_to_tensor(image_list, dtype=tf.string)
    labels = tf.convert_to_tensor(label_list, dtype=tf.int32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=num_epochs,
                                                shuffle=True,
                                                seed=555)

    image, label = read_images_from_disk(input_queue)
    image.set_shape((118, 210, 3))
    image = tf.cast(image, tf.float32)

    # Optional Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation
    image = augment(image)
    # label = tf.one_hot(label, depth=n_class)

    # Optional Image and Label Batching
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              allow_smaller_final_batch=True)
    return image_batch, label_batch


def read_labeled_image_list(image_list_file, label_file, data_root):
    with ExitStack () as stack:
        f_image = stack.enter_context(open(image_list_file, 'r'))
        f_label = stack.enter_context(open(label_file, 'r'))

        image_pathes = []
        labels = []
        for image_file_name, label in zip(f_image, f_label):
            image_file_name = image_file_name.rstrip()
            image_full_path = os.path.join(data_root, image_file_name)
            image_pathes.append(image_full_path)
            labels.append(int(label.rstrip()))
    return image_pathes, labels


def read_images_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    return example, label
