
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    num_classes = 10
    dropout_rate = 0.4
    is_training = True
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    # layer1 - do not use Batch Normalization on the first layer of Discriminator
    conv1 = tf.layers.conv2d(input_layer, 32, [5, 5],
                             strides=[2, 2],
                             padding='same')
    lrelu1 = tf.maximum(0.2 * conv1, conv1)  # leaky relu
    dropout1 = tf.layers.dropout(lrelu1, dropout_rate)

    # layer2
    conv2 = tf.layers.conv2d(dropout1, 64, [3, 3],
                             strides=[2, 2],
                             padding='same')
    batch_norm2 = tf.layers.batch_normalization(conv2, training=is_training)
    lrelu2 = tf.maximum(0.2 * batch_norm2, batch_norm2)

    # layer3
    conv3 = tf.layers.conv2d(lrelu2, 128, [2, 2],
                             strides=[2, 2],
                             padding='same')
    batch_norm3 = tf.layers.batch_normalization(conv3, training=is_training)
    lrelu3 = tf.maximum(0.2 * batch_norm3, batch_norm3)
    dropout3 = tf.layers.dropout(lrelu3, dropout_rate)

    # layer 4
    conv4 = tf.layers.conv2d(dropout3, 128, [2, 2],
                             strides=[2, 2],
                             padding='same')
    # do not use batch_normalization on this layer - next layer, "flatten5",
    # will be used for "Feature Matching"
    lrelu4 = tf.maximum(0.2 * conv4, conv4)

    # layer 5
    flatten_length = lrelu4.get_shape().as_list()[1] * \
                     lrelu4.get_shape().as_list()[2] * lrelu4.get_shape().as_list()[3]
    flatten5 = tf.reshape(lrelu4, (-1, flatten_length))  # used for "Feature Matching"
    fc5 = tf.layers.dense(flatten5, (num_classes))
    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=fc5, units=10)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and eval data
    fraction_labels = 0.2

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    nbr_data_points = train_labels.size
    random_logics = np.random.uniform(size=nbr_data_points) < fraction_labels
    train_data = train_data[random_logics, :]
    train_labels = train_labels[random_logics]


    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=512,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1500,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
  tf.app.run()