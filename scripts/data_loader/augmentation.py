import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops

import random
from random import randint


rand_seed = 555
random.seed(555)
scaling_factor = [0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5]
shift_factor = [0 , 16, 32, 48, 64]

def augment(image, multiple):
    xh, xw, xch = image.get_shape().as_list()

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image, seed=rand_seed)
    image = tf.image.random_flip_up_down(image, seed=rand_seed)

    image = tf.image.random_contrast(image, lower=0.2, upper=1.8, seed=rand_seed)
    image = tf.image.random_brightness(image, max_delta=63, seed=rand_seed)

    # k = randint(0, 3)
    # # rot90 has bug. now skip rot.
    # # rot90 is not affine transformation.
    # image = rot90(image, k=k)

    # # offset image
    # image = shift(image, h, w)

    # image = scaling(image, multiple, h, w, ch)
    # h, w, ch = image.get_shape().as_list()
    # print(h, w, ch)

    return image


def scaling(image, multiple, h, w, ch):
    # h, w, ch = image.get_shape().as_list()
    scale = randint(0, len(scaling_factor)-1)
    new_sz = [int(h*scale)+1, int(w*scale)+1]

    # scaling
    image = tf.image.resize_images(image, new_sz)
    xh, xw, ch = image.get_shape().as_list()
    m0, m1 = xh % multiple, xw % multiple
    d0, d1 = randint(0, m0), randint(0, m1)
    image = tf.random_crop(image, [xh-m0, xw-m1, ch], seed=rand_seed)
    return image


def shift(image, h, w):
    x_shift_idx = randint(0, len(shift_factor)-1)
    y_shift_idx = randint(0, len(shift_factor)-1)
    x_shift = shift_factor[x_shift_idx]
    y_shift = shift_factor[y_shift_idx]
    print(x_shift, y_shift, h, w)
    image = tf.image.pad_to_bounding_box(image, x_shift, y_shift, h-x_shift, w-y_shift)
    return image


def rot90(image, k=1, name=None):
    """Rotate an image counter-clockwise by 90 degrees.
    Args:
    image: A 3-D tensor of shape `[height, width, channels]`.
    k: A scalar integer. The number of times the image is rotated by 90 degrees.
    name: A name for this operation (optional).
    Returns:
    A rotated 3-D tensor of the same type and shape as `image`.
    """
    with ops.name_scope(name, 'rot90', [image, k]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        _Check3DImage(image, require_static=False)
        k = ops.convert_to_tensor(k, dtype=dtypes.int32, name='k')
        k.get_shape().assert_has_rank(0)
        k = math_ops.mod(k, 4)

        def _rot90():
          return array_ops.transpose(array_ops.reverse_v2(image, [1]),
                                     [1, 0, 2])
        def _rot180():
          return array_ops.reverse_v2(image, [0, 1])
        def _rot270():
          return array_ops.reverse_v2(array_ops.transpose(image, [1, 0, 2]),
                                      [1])
        cases = [(math_ops.equal(k, 1), _rot90),
                 (math_ops.equal(k, 2), _rot180),
                 (math_ops.equal(k, 3), _rot270)]

        ret = control_flow_ops.case(cases, default=lambda: image, exclusive=True,
                                    name=scope)

        h, w, ch = image.get_shape().as_list()
        if k==1 or k==3:
            shape = [w, h]
        else:
            shape = [h, w]

        ret.set_shape(shape+[ch])
        return ret


def _Check3DImage(image, require_static=True):
    """Assert that we are working with properly shaped image.
    Args:
    image: 3-D Tensor of shape [height, width, channels]
    require_static: If `True`, requires that all dimensions of `image` are
      known and non-zero.
    Raises:
    ValueError: if `image.shape` is not a 3-vector.
    Returns:
    An empty list, if `image` has fully defined dimensions. Otherwise, a list
    containing an assert op is returned.
    """
    try:
        image_shape = image.get_shape().with_rank(3)
    except ValueError:
        raise ValueError("'image' (shape %s) must be three-dimensional." %
                         image.shape)
    if require_static and not image_shape.is_fully_defined():
        raise ValueError("'image' (shape %s) must be fully defined." %
                         image_shape)
    if any(x == 0 for x in image_shape):
        raise ValueError("all dims of 'image.shape' must be > 0: %s" %
                         image_shape)
    if not image_shape.is_fully_defined():
        return [check_ops.assert_positive(array_ops.shape(image),
                                          ["all dims of 'image.shape' "
                                           "must be > 0."])]
    else:
        return []



from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform


def rotate(X, intensity):
    for i in range(X.shape[0])):
        delta = 30. * intensity # scale using augmentation intensity
        X[i] = rotate(X[i], random.uniform(-delta, delta), mode = 'edge')
    return X


def apply_projection_transform(X, intensity):
    image_size = X.shape[1]
    d = image_size * 0.3 * intensity
    for i in range(X.shape[0])):
        tl_top = random.uniform(-d, d)     # Top left corner, top margin
        tl_left = random.uniform(-d, d)    # Top left corner, left margin
        bl_bottom = random.uniform(-d, d)  # Bottom left corner, bottom margin
        bl_left = random.uniform(-d, d)    # Bottom left corner, left margin
        tr_top = random.uniform(-d, d)     # Top right corner, top margin
        tr_right = random.uniform(-d, d)   # Top right corner, right margin
        br_bottom = random.uniform(-d, d)  # Bottom right corner, bottom margin
        br_right = random.uniform(-d, d)   # Bottom right corner, right margin

        transform = ProjectiveTransform()
        transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))
        X[i] = warp(X[i], transform, output_shape=(image_size, image_size), order = 1, mode = 'edge')

    return X
