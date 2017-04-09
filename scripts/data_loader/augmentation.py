import tensorflow as tf


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
