"""
All tensorflow objects, if not otherwise specified, should be explicity
created with tf.float32 datatypes. Not specifying this datatype for variables and
placeholders will cause your code to fail some tests.

You do not need to import any other libraries for this assignment.

Along with the provided functional prototypes, there is another file,
"train.py" which calls the functions listed in this file. It trains the
specified network on the MNIST dataset, and then optimizes the loss using a
standard gradient decent optimizer. You can run this code to check the models
you create in part II.
"""

import tensorflow as tf
import time

""" PART I """


def add_consts():
    """
    EXAMPLE:
    Construct a TensorFlow graph that declares 3 constants, 5.1, 1.0 and 5.9
    and adds these together, returning the resulting tensor.
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.constant(5.9)
    a1 = tf.add(c1, c2)
    af = tf.add(a1, c3)
    return af


def add_consts_with_placeholder():
    """ 
    Construct a TensorFlow graph that constructs 2 constants, 5.1, 1.0 and one
    TensorFlow placeholder of type tf.float32 that accepts a scalar input,
    and adds these three values together, returning as a tuple, and in the
    following order:
    (the resulting tensor, the constructed placeholder).
    """
    c1 = tf.constant(5.1)
    c2 = tf.constant(1.0)
    c3 = tf.placeholder(tf.float32)
    a1 = tf.add(c1, c2)
    af = tf.add(a1, c3)    

    return af, c3


def my_relu(in_value):
    """
    Implement a ReLU activation function that takes a scalar tf.placeholder as input
    and returns the appropriate output. For more information see the assignment spec.
    """
    c1 = tf.placeholder(tf.float32)
    c2 = tf.constant(0.0)
    out_value = tf.maximum(c2,c1)
    
    return out_value


def my_perceptron(x):

    i = tf.placeholder(tf.float32, shape=[x])

    weight_1 = tf.get_variable("weight_1",shape=(),initializer=tf.ones_initializer,trainable=True)    
    #init = tf.global_variables_initializer()
    #sess.run(init)  
    out =  my_relu(tf.matmul(i, weight_1))
    #[i,out] = my_perceptron(3)
    #print(sess.run(out, feed_dict={i:[1,2,3]}))


    #does matmul produce a single value?
    return i, out


""" PART II """
fc_count = 0  # count of fully connected layers. Do not remove.


def input_placeholder(n_features):
    return tf.placeholder(dtype=tf.float32, shape=[None, n_features],
                          name="image_input")


def target_placeholder(n_classes):
    return tf.placeholder(dtype=tf.float32, shape=[None, n_classes],
                          name="image_target_onehot")




def onelayer(X, Y, layersize=10):
    """
    Create a Tensorflow model for logistic regression (i.e. single layer NN)

    :param X: The input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned  (variables in the
    python sense, not in the Tensorflow sense, although some may be
    Tensorflow variables). They must be returned in the following order.
        w: Connection weights
        b: Biases
        logits: The input to the activation function
        preds: The output of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    
  

    w = tf.get_variable("weight",shape=[X.get_shape().as_list()[1], 10],initializer=tf.ones_initializer)
    b = tf.get_variable("bias", shape=[10], initializer=tf.constant_initializer(0.1))    

    logits = tf.matmul(X,w) + b
    preds = tf.nn.softmax(logits)
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(batch_xentropy)

    batch_loss = tf.reduce_mean(batch_xentropy, name='batch_xentropy')
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    return w, b, logits, preds, batch_xentropy, batch_loss


def twolayer(X, Y, hiddensize=30, outputsize=10):
    """
    Create a Tensorflow model for a Neural Network with one hidden layer

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        W1: Connection weights for the first layer
        b1: Biases for the first layer
        W2: Connection weights for the second layer
        b2: Biases for the second layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch
    """
    w1 = tf.get_variable("weight_1",shape=[X.get_shape().as_list()[1], outputsize],initializer=tf.ones_initializer)
    w2 = tf.get_variable("weight_2",shape=[outputsize],initializer=tf.ones_initializer)    
    b1 = tf.get_variable("bias_1", shape=[outputsize], initializer=tf.constant_initializer(0.1)) 
    b2 = tf.get_variable("bias_2", shape=[outputsize], initializer=tf.constant_initializer(0.1))
    temp_logits = tf.matmul(X,w1) + b1
    temp_preds = tf.nn.softmax(temp_logits)
    logits = temp_preds*w2 + b2
    preds = tf.nn.softmax(logits)

    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y)
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(batch_xentropy)

    batch_loss = tf.reduce_mean(batch_xentropy, name='batch_xentropy')
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    return w1, b1, w2, b2, logits, preds, batch_xentropy, batch_loss


def convnet(X, Y, convlayer_sizes=[10, 10], \
            filter_shape=[3, 3], outputsize=10, padding="same", reshape_size=7840):
    """
    Create a Tensorflow model for a Convolutional Neural Network. The network
    should be of the following structure:
    conv_layer1 -> conv_layer2 -> fully-connected -> output

    :param X: The  input placeholder for images from the MNIST dataset
    :param Y: The output placeholder for image labels
    :return: The following variables should be returned in the following order.
        conv1: A convolutional layer of convlayer_sizes[0] filters of shape filter_shape
        conv2: A convolutional layer of convlayer_sizes[1] filters of shape filter_shape
        w: Connection weights for final layer
        b: biases for final layer
        logits: The inputs to the activation function
        preds: The outputs of the activation function (a probability
        distribution over the 10 digits)
        batch_xentropy: The cross-entropy loss for each image in the batch
        batch_loss: The average cross-entropy loss of the batch

    hints:
    1) consider tf.layer.conv2d
    2) the final layer is very similar to the onelayer network. Only the input
    will be from the conv2 layer. If you reshape the conv2 output using tf.reshape,
    you should be able to call onelayer() to get the final layer of your network
    """
    conv1 = tf.layers.conv2d(
      inputs=X,
      filters=convlayer_sizes[0],
      kernel_size=filter_shape,
      strides=(1,1),
      padding="same",
      activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=convlayer_sizes[1],
      kernel_size=filter_shape,
      strides=(1,1),
      padding="same",
      activation=tf.nn.relu)


    reshaped_conv2 = tf.reshape(conv2, [-1, reshape_size])
    #output_flat = tf.reshape(Y, [-1, 1])

    #final_shape = tf.gather(reshaped_conv2, int(reshaped_conv2.get_shape()[0]) - 1)
    out_size = outputsize

    logits = tf.contrib.layers.fully_connected(reshaped_conv2, out_size, activation_fn=None)
    preds = tf.nn.softmax(logits)

    cost = tf.losses.softmax_cross_entropy(Y, logits)
    loss = tf.reduce_mean(cost, name='loss')


    #w, b, logits, preds, batch_xentropy, batch_loss = onelayer(reshaped_conv2,output_flat)
    return logits, preds, loss


def train_step(sess, batch, X, Y, train_op, loss_op, summaries_op):
    """
    Run one step of training.

    :param sess: the current session
    :param batch: holds the inputs and target outputs for the current minibatch
    batch[0] - array of shape [minibatch_size, 784] with each row holding the
    input images
    batch[1] - array of shape [minibatch_size, 10] with each row holding the
    one-hot encoded targets
    :param X: the input placeholder
    :param Y: the output target placeholder
    :param train_op: the tensorflow operation that will run one step of training
    :param loss_op: the tensorflow operation that will return the loss of your
    model on the batch input/output

    :return: a 3-tuple: train_op_result, loss, summary
    which are the results of running the train_op, loss_op and summaries_op
    respectively.
    """
    train_result, loss, summary = \
        sess.run([train_op, loss_op, summaries_op], feed_dict={X: batch[0], Y: batch[1]})
    return train_result, loss, summary



def accuracy(sess, dataset, batch_size, X, Y, accuracy_op):
    # compute number of batches for given batch_size
    num_test_batches = dataset.num_examples // batch_size

    overall_accuracy = 0.0
    for i in range(num_test_batches):
        batch = dataset.next_batch(batch_size)
        accuracy_batch = \
            sess.run(accuracy_op, feed_dict={X: batch[0], Y: batch[1]})
        overall_accuracy += accuracy_batch

    return overall_accuracy / num_test_batches


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name + '_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)



def nn_train(sess, curr_dataset, n_training_epochs, batch_size,
          summaries_op, accuracy_summary_op, train_writer, test_writer,
          X, Y, train_op, loss_op, accuracy_op, accuracy_placeholder):
    # compute number of batches for given batch_size
    num_train_batches = curr_dataset.train.num_examples // batch_size

    # record starting time
    train_start = time.time()

    # Run through the entire dataset n_training_epochs times
    for i in range(n_training_epochs):
        # Initialise statistics
        training_loss = 0
        epoch_start = time.time()

        # Run the SGD train op for each minibatch
        for _ in range(num_train_batches):
            batch = curr_dataset.train.next_batch(batch_size)
            trainstep_result, batch_loss, summary = train_step(sess, batch, X, Y, train_op, loss_op, summaries_op)
            train_writer.add_summary(summary, i)
            training_loss += batch_loss

        # Timing and statistics
        epoch_duration = round(time.time() - epoch_start, 2)
        ave_train_loss = training_loss / num_train_batches

        # Get accuracy
        train_accuracy = \
            accuracy(sess, curr_dataset.train, batch_size, X, Y, accuracy_op)
        test_accuracy = \
            accuracy(sess, curr_dataset.test, batch_size, X, Y, accuracy_op)

        # log accuracy at the current epoch on training and test sets
        train_acc_summary = sess.run(accuracy_summary_op,
                                     feed_dict={accuracy_placeholder: train_accuracy})
        train_writer.add_summary(train_acc_summary, i)
        test_acc_summary = sess.run(accuracy_summary_op,
                                    feed_dict={accuracy_placeholder: test_accuracy})
        test_writer.add_summary(test_acc_summary, i)
        [writer.flush() for writer in [train_writer, test_writer]]

        train_duration = round(time.time() - train_start, 2)

        # Output to montior training
        print('Epoch {0}, Training Loss: {1}, Training accuracy: {2}, Test accuracy: {3}, \
                time: {4}s, total time: {5}s'.format(i, ave_train_loss, train_accuracy,
                                                     test_accuracy, epoch_duration,
                                                     train_duration))
    print('Total training time: {0}s'.format(train_duration))
    return train_accuracy, test_accuracy


def get_accuracy_op(preds_op, Y):
    with tf.name_scope('accuracy_ops'):
        correct_preds_op = tf.equal(tf.argmax(preds_op, 1), tf.argmax(Y, 1))
        # the tf.cast sets True to 1.0, and False to 0.0. With N predictions, of
        # which M are correct, the mean will be M/N, i.e. the accuracy
        accuracy_op = tf.reduce_mean(tf.cast(correct_preds_op, tf.float32))
    return accuracy_op
