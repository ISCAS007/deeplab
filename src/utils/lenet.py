# -*- coding: utf-8 -*-
import functools
import tensorflow as tf
slim = tf.contrib.slim

def lenet(images, num_classes=10, is_training=False,
          dropout_keep_prob=0.5,
          prediction_fn=slim.softmax,
          scope='LeNet'):
    """Creates a variant of the LeNet model.

    Note that since the output is a set of 'logits', the values fall in the
    interval of (-infinity, infinity). Consequently, to convert the outputs to a
    probability distribution over the characters, one will need to convert them
    using the softmax function:

          logits = lenet.lenet(images, is_training=False)
          probabilities = tf.nn.softmax(logits)
          predictions = tf.argmax(logits, 1)

    Args:
      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset. If 0 or None, the logits
        layer is omitted and the input features to the logits layer are returned
        instead.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
       net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
        is a non-zero integer, or the inon-dropped-out nput to the logits layer
        if num_classes is 0 or None.
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    end_points = {}

    with tf.variable_scope(scope, 'LeNet', [images]):
        net = end_points['conv1'] = slim.conv2d(
            images, 32, [5, 5], scope='conv1')
        net_branch1 = end_points['branch1'] = slim.conv2d(
                net, 32, [3,3], scope='branch1')
        net_branch2 = end_points['branch2'] = slim.conv2d(
                net, 32, [3,3], scope='branch2')
        # Create a variable.
        with tf.variable_scope('branch'):
            w1 = tf.Variable(0.5, name='w1')
            w2 = tf.Variable(0.5, name='w2')
        
        net=tf.sigmoid(w1)*net_branch1+tf.sigmoid(w2)*net_branch2
        net = end_points['pool1'] = slim.max_pool2d(
            net, [2, 2], 2, scope='pool1')
        net = end_points['conv2'] = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = end_points['pool2'] = slim.max_pool2d(
            net, [2, 2], 2, scope='pool2')
        net = slim.flatten(net)
        end_points['Flatten'] = net

        net = end_points['fc3'] = slim.fully_connected(net, 1024, scope='fc3')
        if not num_classes:
            return net, end_points
        net = end_points['dropout3'] = slim.dropout(
            net, dropout_keep_prob, is_training=is_training, scope='dropout3')
        logits = end_points['Logits'] = slim.fully_connected(
            net, num_classes, activation_fn=None, scope='fc4')

    end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

    return logits, end_points


lenet.default_image_size = 28


def lenet_arg_scope(weight_decay=0.0):
    """Defines the default lenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
            activation_fn=tf.nn.relu) as sc:
        return sc
    
def get_network_fn(num_classes, weight_decay=0.0, is_training=False):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification. If 0 or None,
        the logits layer is omitted and its input features are returned instead.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature:
            net, end_points = network_fn(images)
        The `images` input is a tensor of shape [batch_size, height, width, 3]
        with height = width = network_fn.default_image_size. (The permissibility
        and treatment of other sizes depends on the network_fn.)
        The returned `end_points` are a dictionary of intermediate activations.
        The returned `net` is the topmost layer, depending on `num_classes`:
        If `num_classes` was a non-zero integer, `net` is a logits tensor
        of shape [batch_size, num_classes].
        If `num_classes` was 0 or `None`, `net` is a tensor with the input
        to the logits layer of shape [batch_size, 1, 1, num_features] or
        [batch_size, num_features]. Dropout has not been applied to this
        (even if the network's original classification does); it remains for
        the caller to do this or not.

    Raises:
      ValueError: If network `name` is not recognized.
    """
    func = lenet

    @functools.wraps(func)
    def network_fn(images, **kwargs):
        arg_scope = lenet_arg_scope(weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training, **kwargs)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn