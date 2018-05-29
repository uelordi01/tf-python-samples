import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

input_size = 784
no_classes = 10
batch_size = 100
total_batches = 200

# x_input is the input image
x_input = tf.placeholder(tf.float32, shape=[None, input_size])
y_input = tf.placeholder(tf.float32, shape=[None, no_classes])

def add_variable_summary(tf_variable, summary_name):
    with tf.name_scope(summary_name + '_summary'):
        mean = tf.reduce_mean(tf_variable)
        tf.summary.scalar('Mean', mean)
        with tf.name_scope('standard_deviation'):
            standard_deviation = tf.sqrt(tf.reduce_mean(
            tf.square(tf_variable - mean)))
            tf.summary.scalar('StandardDeviation', standard_deviation)
            tf.summary.scalar('Maximum', tf.reduce_max(tf_variable))
            tf.summary.scalar('Minimum', tf.reduce_min(tf_variable))
            tf.summary.histogram('Histogram', tf_variable)

def convolution_layer(input_layer, filters, kernel_size=[3, 3],
activation=tf.nn.relu):
    layer = tf.layers.conv2d(
    inputs=input_layer,
    filters=filters,
    kernel_size=kernel_size,
    activation=activation,
    )
    add_variable_summary(layer, 'convolution')
    return layer

def pooling_layer(input_layer, pool_size=[2, 2], strides=2):
    layer = tf.layers.max_pooling2d(
    inputs=input_layer,
    pool_size=pool_size,
    strides=strides
    )
    add_variable_summary(layer, 'pooling')
    return layer

def dense_layer(input_layer, units, activation=tf.nn.relu):
    layer = tf.layers.dense(
    inputs=input_layer,
    units=units,
    activation=activation
    )
    add_variable_summary(layer, 'dense')
    return layer

x_input_reshape = tf.reshape(x_input, [-1, 28, 28, 1],
    name='input_reshape')

convolution_layer_1 = convolution_layer(x_input_reshape, 64)

pooling_layer_1 = pooling_layer(convolution_layer_1)

convolution_layer_2 = convolution_layer(pooling_layer_1, 128)

pooling_layer_2 = pooling_layer(convolution_layer_2)

flattened_pool = tf.reshape(pooling_layer_2, [-1, 5 * 5 * 128],
    name='flattened_pool')

dense_layer_bottleneck = dense_layer(flattened_pool, 1024)

x_input_reshape = tf.reshape(x_input, [-1, 28, 28, 1],
    name='input_reshape')

dropout_bool = tf.placeholder(tf.bool)
dropout_layer = tf.layers.dropout(
    inputs=dense_layer_bottleneck,
    rate=0.4,
    training=dropout_bool
)
logits = dense_layer(dropout_layer, no_classes)

with tf.name_scope('loss'):
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=y_input, logits=logits)
    loss_operation = tf.reduce_mean(softmax_cross_entropy, name='loss')
    tf.summary.scalar('loss', loss_operation)