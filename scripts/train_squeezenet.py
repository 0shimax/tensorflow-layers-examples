import sys
sys.path.append("./scripts/data_loader")
sys.path.append("./models")
from mnist_loader import load_data
from squeeze_net import squeeze_net

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


tf.logging.set_verbosity(tf.logging.INFO)
n_class = 10


# define model
def model_fn(features, labels, mode):
    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=n_class)
    predictions, loss, train_op = squeeze_net(input_layer, labels, mode, n_class)

    # return
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
    train_data, train_labels, eval_data, eval_labels = load_data("./data/mnist")

    # create Estimator
    classifier = tf.contrib.learn.Estimator(
        model_fn=model_fn, model_dir="./results/models/squeeze_net")

    # setting log
    tensors_to_log = {"probabilities": "softmax_tensor", "classes": "argmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # learning
    classifier.fit(
        x=train_data,
        y=train_labels,
        batch_size=100,
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
        x=eval_data, y=eval_labels, metrics=metrics
    )

    print(eval_results)

if __name__ == "__main__":
    tf.app.run()
