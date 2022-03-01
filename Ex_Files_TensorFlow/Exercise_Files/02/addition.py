import os
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define computational graph (only 3 nodes for simple example)
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
# NOTE: type is the type of data (float32) and the name is what we see on a graphical model
addition = tf.add(X, Y, name="add") # pulls data from x and y nodes and adds result

#---------------------------------------------------------------

# Create the session object
with tf.Session() as session:
    # passes in arrays for X and Y and outputs array (because tensor flow always works with multi-dimensional arrays aka tensors)
    result = session.run(addition, feed_dict={X: [1, 2, 10], Y: [4, 2, 10]}) # runs addition session with data fed in

    print(result) # result will be "[5.    4.     20.]"
