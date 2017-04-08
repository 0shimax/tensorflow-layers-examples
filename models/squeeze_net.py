import sys
sys.path.append("./scripts/functions")
from spp import spatial_pyramid_pool

import tensorflow as tf


def Fire(x, s1, e1, e3):
    conv1=tf.layers.conv2d(
        inputs=x,
        filters=s1,
        kernel_size=1,
        activation=tf.nn.elu)

    conv2=tf.layers.conv2d(
        inputs=conv1,
        filters=e1,
        kernel_size=1)

    conv3=tf.layers.conv2d(
        inputs=conv1,
        filters=e3,
        kernel_size=3,
        padding="same")

    h_out = tf.concat([conv2, conv3], axis=3)
    return tf.nn.elu(h_out)


def squeeze_net(x, y, mode, n_class):
    conv1=tf.layers.conv2d(
            inputs=x,
            filters=96,
            kernel_size=7,
            strides=2,
            padding='same',
            activation=tf.nn.elu)

    pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=2)

    fire2=Fire(pool1, 16, 64, 64)
    fire3=Fire(fire2, 16, 64, 64)
    fire4=Fire(fire3, 16, 128, 128)

    pool2=tf.layers.max_pooling2d(inputs=fire4, pool_size=3, strides=2)

    fire5=Fire(pool2, 32, 128, 128)
    fire6=Fire(fire5, 48, 192, 192)
    fire7=Fire(fire6, 48, 192, 192)
    fire8=Fire(fire7, 64, 256, 256)

    # pool3=tf.layers.max_pooling2d(inputs=fire8, pool_size=3, strides=2)
    # fire9=Fire(pool3, 64, 256, 256)

    g_pool=tf.contrib.layers.avg_pool2d(inputs=fire8, kernel_size=2)

    logits = tf.layers.dense(
        inputs=g_pool,
        units=n_class)

    # spp = spatial_pyramid_pool(fire8)
    # spp = tf.reshape(spp, [-1, 1, 1, 512*21])
    #
    # conv9=tf.layers.conv2d(
    #         inputs=spp,
    #         filters=4096,
    #         kernel_size=1,
    #         activation=tf.nn.elu)
    #
    # logits=tf.layers.conv2d(
    #         inputs=conv9,
    #         filters=n_class,
    #         kernel_size=1)

    logits = tf.reshape(logits, [-1, n_class])

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != tf.contrib.learn.ModeKeys.INFER:
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD"
        )

    # Generate Predictions
    # you can obserb values while running trainig.
    predictions = {
        "classes": tf.argmax(input=logits, axis=1, name="argmax_tensor"),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
    }

    return predictions, loss, train_op
