import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)


input_size = 784
no_classes = 10
batch_size = 100
total_batches = 200


# x_input is the input image
x_input = tf.placeholder(tf.float32, shape=[None, input_size])
y_input = tf.placeholder(tf.float32, shape=[None, no_classes])

# defining fully connected layers:
weights = tf.Variable(tf.random_normal([input_size, no_classes]))
# variable_summaries(weights)

bias = tf.Variable(tf.random_normal([no_classes]))
variable_summaries(bias)

logits = tf.matmul(x_input, weights) + bias

# The logits produced by the perceptron has to be compared against one-hot labels
# y_input . As learned in Chapter 1 , Getting Started, it is better to use softmax coupled with
# cross-entropy for comparing logits and one-hot labels.

softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
labels=y_input, logits=logits)
# we compute the loss function
loss_operation = tf.reduce_mean(softmax_cross_entropy)
tf.summary.scalar("loss", loss_operation)
# variable_summaries(loss_operation)
# we apply the backpropagation through gradient descent to the loss function

optimiser = tf.train.GradientDescentOptimizer(
learning_rate=0.5).minimize(loss_operation)

# Now you have defined the model and training operation. The next step is to start training
# the model with the data. During training, the gradients are calculated and the weights are
# updated. The variables have not yet been initialized. Next, start the session and initialize the
# variables using a global variable initializer:


summ = tf.summary.merge_all()


session = tf.Session()
train_writer = tf.summary.FileWriter( 'train',
                                     session.graph)
session.run(tf.global_variables_initializer())
counter = 0;
for batch_no in range(total_batches):
    mnist_batch = mnist_data.train.next_batch(batch_size)
    loss_value  = session.run([optimiser, loss_operation], feed_dict={
    x_input: mnist_batch[0],
    y_input: mnist_batch[1]
    })
    summary = session.run(summ, feed_dict={
    x_input: mnist_batch[0],
    y_input: mnist_batch[1]})
    print(loss_value)
    train_writer.add_summary(summary, counter)
    counter = counter +1
# for batch_no in range(total_batches)
#     mnist_batch = mnist_data.train.next_batch(batch_size),
#
predictions = tf.argmax(logits, 1)
correct_predictions = tf.equal(predictions, tf.argmax(y_input, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_predictions,
tf.float32))
test_images, test_labels = mnist_data.test.images, mnist_data.test.labels
accuracy_value = session.run(accuracy_operation, feed_dict={
x_input: test_images,
y_input: test_labels
})

print('Accuracy : ', accuracy_value)
session.close()
