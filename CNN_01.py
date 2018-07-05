from __future__ import print_function
import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
#one_hot encoding is to make the data more machine readable

#hyper parameters

learning_rate=0.01
training_iters=200000
batch_size=128#samples
display_step=10 #every 10 iteration, display

#network paras
n_input=784#image shape is gpoing to be 28*28
n_classes=10 #10 digits 0~9
dropout=0.75 #prevents overfitting by randomly turning off some neurons during training,so the data forced to find new paths between layers to generalize the model
#ex: old people explanation

x=tf.placeholder(tf.float32,[None,n_input])#images
y=tf.placeholder(tf.float32,[None,n_classes])#labels
keep_prob=tf.placeholder(tf.float32)

#creating the C layers

def conv2d(x,w,b,strides=1):
	x=tf.nn.conv2d(x,w,strides=[1,strides,strides,1], padding='SAME')
	x =tf.nn.bias_add(x, b)
	return(tf.nn.relu(x))
#convolution is tranforming it o some way(putting filters)
#must have some thing about original image
#bias makes the model more accurate
#strides=list of ints / tensors=data

def maxpool2d(x,k=2):#pooling=small rectangular blocks from the conv layer and sub samples them little pools from the image.
	return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],padding="SAME")


#creating the model

def conv_net(x,weights,biases,dropout):
	x=tf.reshape(x,shape=[-1,28,28,1])

	#conv layer
	conv1=conv2d(x,weights["wc1"],biases["bc1"])
	#max pooling layer
	conv1=maxpool2d(conv1,k=2)

	conv2=conv2d(conv1,weights["wc2"],biases["bc2"])
	conv2=maxpool2d(conv2,k=2)

	#fully connected layer
	fc1=tf.reshape(conv2,[-1,weights["wd1"].get_shape().as_list()[0]])
	#matrix mul
	fc1=tf.add(tf.matmul(fc1,weights["wd1"]),biases["bd1"])
	fc1=tf.nn.relu(fc1)

	#applying dropout
	fc1=tf.nn.dropout(fc1,dropout)

	#output is going to predict our class
	out=tf.add(tf.matmul(fc1,weights["out"]),biases["out"])
	return out

#creating weights
weights={
"wc1":tf.Variable(tf.random_normal([5,5,1,32])),#5*5,1=inputs,32=bits 
"wc2":tf.Variable(tf.random_normal([5,5,32,64])),#5*5,32=inputs,64=bits 
"wd1":tf.Variable(tf.random_normal([7*7*64,1024])),#7*7*64=inputs,1024=bits /outputs
"out":tf.Variable(tf.random_normal([1024,n_classes]))#1024=inputs,number of classes=10
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
#construct model
pred=conv_net(x,weights,biases,keep_prob)#keep_prob=dropout

#deine optimizer and loss
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))#measuring the probability error in a classification task /mutually exclusive one
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()#everytime initializing a tf graph, must initialize variables

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iterations " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Completed!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))
