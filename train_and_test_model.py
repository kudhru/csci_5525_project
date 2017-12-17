import sys

import numpy as np
import tensorflow as tf

from utils import generate_one_hot_num_array, read_data


def _get_batch_data(X_train, Y_train, batch_size, iteration):
    start_index = iteration * batch_size
    end_index = start_index + batch_size
    X_batch = X_train[start_index:end_index]
    Y_batch = Y_train[start_index:end_index]
    return X_batch, Y_batch


def train_and_test( X_train, Y_train, X_test, Y_test, batch_size = 1000, learning_rate=0.5, n_epochs = 1000):
    batch_size = min(min(len(X_train), len(X_test)), batch_size)
    D = len(X_train[0])
    num_class = len(np.unique(np.append(Y_train, Y_test)))

    x = tf.placeholder(tf.float32, [batch_size, D])
    y = tf.placeholder(tf.float32, [batch_size, num_class])


    W = tf.Variable(tf.zeros([D, num_class]))
    b = tf.Variable(tf.zeros([num_class]))


    pred_y = tf.matmul(x, W) + b
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred_y))

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    print 'Optimization starting!'

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        n_batches = int(len(X_train) / batch_size)
        for iter in range(n_epochs):  # train the model n_epochs times
            # print iter
            total_loss = 0
            for j in range(n_batches):
                # print j
                X_batch, Y_batch = _get_batch_data(X_train, Y_train, batch_size, j)
                Y_batch = generate_one_hot_num_array(Y_batch, num_class)
                curr_step, curr_entropy = sess.run([train_step, cross_entropy],
                                                            feed_dict={x: X_batch, y: Y_batch})
                total_loss += curr_entropy

                if j % 100 == 0:
                    print 'Average loss epoch {0} {1}: {2}'.format(iter, j, total_loss / (j+1))

            print 'Average loss epoch {0}: {1}'.format(iter, total_loss / n_batches)

        print 'Optimization Finished!'  # should be around 0.35 after 25 epochs

        # test the model
        preds = tf.nn.softmax(pred_y)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))  # need numpy.count_nonzero(boolarr) :(

        n_batches = int(len(X_test) / batch_size)
        total_correct_preds = 0

        for iter in range(n_batches):
            X_batch, Y_batch = _get_batch_data(X_test, Y_test, batch_size, iter)
            Y_batch = generate_one_hot_num_array(Y_batch, num_class)
            accuracy_batch = sess.run(accuracy, feed_dict={x: X_batch, y: Y_batch})
            total_correct_preds += accuracy_batch

        print 'Accuracy {0}'.format(total_correct_preds / len(X_test))

train_file = sys.argv[1]
test_file = sys.argv[2]

train_images, train_labels = read_data(train_file)
test_images, test_labels = read_data(test_file)

print len(train_images)
print len(train_labels)
print train_images[1].shape
print train_labels[1]

print len(test_images)
print len(test_labels)
print test_images[1].shape
print test_labels[1]

train_and_test(train_images, train_labels, test_images, test_labels, batch_size=1000, n_epochs=100, learning_rate=0.1)

    
    
    
    
    

    
    
            
        





    

    
