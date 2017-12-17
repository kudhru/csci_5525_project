import numpy as np
import random
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

from generate_one_hot import generate_one_hot_num_array


def read_data(data_file):
    data = np.loadtxt(data_file, delimiter=',')
    dimension = data.shape[1] - 1

    target = data[:, 0]
    data_without_target = data[:, 1:dimension + 1]
    data_without_target.astype(float)
    return data_without_target, target


def generate_next_batch(image_pool, labels,batch_size, feat):
    batch_data = np.zeros(shape = (batch_size,feat))
    batch_labels = np.zeros(shape = (batch_size,2))
    counter = 0

    for i in range(batch_size):
        ind = random.randint(1,len(image_pool))

        batch_data[i] = image_pool[ind]
        batch_labels[i] = labels[ind]

    return (batch_data, batch_labels)

def _get_batch_data(X_train, Y_train, batch_size):
    train_size = batch_size / float(len(X_train))
    if train_size < 1.0:
        X_batch, _1, Y_batch, _2 = train_test_split(
            X_train,
            Y_train,
            train_size=batch_size / float(len(X_train))
        )
        actual_size = len(X_batch)
        diff = batch_size - actual_size
        for _ in range(diff):
            index = np.random.randint(len(X_train))
            X_batch = np.append(X_batch, [X_train[index]], axis=0)
            Y_batch = np.append(Y_batch, [Y_train[index]], axis=0)
    else:
        X_batch, Y_batch = X_train, Y_train
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
                X_batch, Y_batch = _get_batch_data(X_train, Y_train, batch_size)
                Y_batch = generate_one_hot_num_array(Y_batch, num_class)
                # Y_batch = np.array([[1.,0.] if Y_batch[i] == 1 else [0.,1.] for i in range(Y_batch.shape[0])])
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
            X_batch, Y_batch = _get_batch_data(X_test, Y_test, batch_size)
            Y_batch = generate_one_hot_num_array(Y_batch, num_class)
            # Y_batch = np.array([[1., 0.] if Y_batch[i] == 1 else [0., 1.] for i in range(Y_batch.shape[0])])
            accuracy_batch = sess.run(accuracy, feed_dict={x: X_batch, y: Y_batch})
            total_correct_preds += accuracy_batch

        print 'Accuracy {0}'.format(total_correct_preds / len(X_test))

    # return(sess.run(W))


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

train_and_test(train_images, train_labels, test_images, test_labels, batch_size=1000, n_epochs=5, learning_rate=0.1)

    
    
    
    
    

    
    
            
        





    

    
