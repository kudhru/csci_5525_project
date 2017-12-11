from scipy import misc
import scipy
import numpy as np
import random

import tensorflow as tf


train_images = {}#np.zeros(shape = (10000,230400))
train_labels = {}#np.zeros(shape = (10000,2))


with open("train/labels.csv","r") as f:
    
    for l in f:
        
        label = np.zeros(shape = (2,))

        
        a = l.split(",")
        
        if int(a[1]) == 1:
            label[0] = int(0)
            label[1] = int(1)
        
            train_labels[int(a[0])] = label

        else:
            label[0] = int(1)
            label[1] = int(0)
        
            train_labels[int(a[0])] = label
    
        
                
counter = 1

for _ in range(10000):
    filename = "train/train"+str(counter)+".png"
    
    image = scipy.misc.imread(filename, flatten=True)
    image = image.flatten()
    #print(image.shape)
    train_images[counter] = image

    counter = counter+1


print(train_images[4630].shape)
print(train_labels[4631])



#READING OF THE TEST SET

test_labels = {}#np.zeros(shape = (2000,2))
test_images = {}#np.zeros(shape = (2000,230400))


with open("test/labels.csv","r") as f:

    for l in f:
        
        label = np.zeros(shape = (2,))
        
        a = l.split(",")

        if int(a[1]) == 1:
            label[0] = int(0)
            label[1] = int(1)
            
            test_labels[int(a[0])] = label

        else:
            
            label[0] = int(1)
            label[1] = int(0)
            
            test_labels[int(a[0])] = label
        
                         
                
counter = 1

for _ in range(2000):
    filename = "test/test"+str(counter)+".png"
    
    image = scipy.misc.imread(filename, flatten=True)
    image = image.flatten()
    
    test_images[counter] = image

    counter = counter+1


print(test_images[1].shape)
print(test_labels[1])




print("DONE!")



def generate_next_batch( image_pool, labels,batch_size, feat):

    
    batch_data = np.zeros(shape = (batch_size,feat))
    batch_labels = np.zeros(shape = (batch_size,2))
    counter = 0

    for i in range(batch_size):
        ind = random.randint(1,len(image_pool))

        batch_data[i] = image_pool[ind]
        batch_labels[i] = labels[ind]

    return (batch_data, batch_labels)

X,Y = generate_next_batch(train_images, train_labels,10,len(train_images[1]))

print(X.shape)
print(Y.shape)


def train( X_train, Y_train, batch_size = 1000):

    D = len(X_train[1])
    
    x = tf.placeholder(tf.float32,[batch_size,D])
    y_ = tf.placeholder(tf.float32, [batch_size,2])

    W = tf.Variable(tf.zeros([D,2]))
    b = tf.Variable(tf.zeros([2]))

    a = tf.sigmoid(tf.add(tf.matmul(x,W),b))

    cross_entropy = tf.reduce_mean(-(y_ * tf.log(a) + (1-y_) * tf.log(1-a)))
    
    z = tf.matmul(x,W)

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    for epoch in range(1000):
        #print("Length is ",len(X_train))
        #print("Length of batch ",batch_size)
        idx = np.random.choice(len(X_train), batch_size, replace = False)
        _,l = sess.run([train_step,cross_entropy], feed_dict={x:X_train[idx], y_: Y_train[idx]})
        

        if epoch%100 == 0:
            
            print("Loss: "+str(l))

    return(sess.run(W))


w = train(X,Y,10)

    
    
    
    
    

    
    
            
        





    

    
