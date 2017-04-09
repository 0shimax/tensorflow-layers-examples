import sys, os
sys.path.append("./scripts/data_loader")
sys.path.append("./models")
from mnist_loader import load_data
from squeeze_net import squeeze_net
from preprocessor_example import file_input_fn, mnist_batch_input_fn, image_path_getter, label_op_getter, minibatch_loader

import functools
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


tf.logging.set_verbosity(tf.logging.INFO)
n_class = 10


# define model
def model_fn(features, labels, mode):
    # you need to convert labels to one_hot for accuracy metrics.
    # (input of accuracy metrics must be vector of scala labels.)
    labels = tf.one_hot(labels, depth=n_class)
    predictions, loss, train_op = squeeze_net(features, labels, mode, n_class)

    # return
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
    # train_data, train_labels, eval_data, eval_labels = load_data("./data/mnist")
    mnist = load_data("./data/mnist")

    # create Estimator
    run_config = tf.contrib.learn.RunConfig(save_summary_steps=10)
    classifier = tf.contrib.learn.Estimator(
        model_fn=model_fn, model_dir="./results/models/squeeze_net", config=run_config)

    # setting log
    tensors_to_log = {"probabilities": "softmax_tensor", "classes": "argmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    img_pointer_name = 'image_pointer'
    label_file_name = 'labels'
    data_root = './data'
    img_pointer_path = os.path.join(data_root, img_pointer_name)
    label_file_path = os.path.join(data_root, label_file_name)

    img_files = image_path_getter('image_pointer')
    op_labels = label_op_getter('labels')
    # learning
    classifier.fit(
        # input_fn=lambda: mnist_batch_input_fn(mnist[:2]),
        input_fn=lambda: minibatch_loader( \
            img_pointer_path, label_file_path, data_root, \
            n_class=10, batch_size=100, num_epochs=1),
        steps=20,
        monitors=[logging_hook]
    )

    metrics = {
        "accuracy":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes",
        )
    }

    # evaluate
    eval_results = classifier.evaluate(
        # input_fn=lambda: mnist_batch_input_fn(mnist[2:]),
        input_fn=lambda: minibatch_loader( \
            img_pointer_path, label_file_path, data_root, \
            n_class=10, batch_size=1, num_epochs=1),
        metrics=metrics
    )

    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
