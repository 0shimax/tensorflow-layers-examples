import os, sys
sys.path.append("./scripts/data_loader")
from augmentation import augment

import tensorflow as tf
import numpy as np
from contextlib import ExitStack


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
    image = augment(image, 8)

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
    example = tf.image.decode_image(file_contents, channels=3)
    return example, label
