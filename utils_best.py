import scipy.stats as st
import tensorflow as tf
import numpy as np
import sys

def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis = 2)
    return out_filter

def blur(x):
    kernel_var = gauss_kernel(21, 3, 3)
    return tf.nn.depthwise_conv2d(x, kernel_var, [1, 1, 1, 1], padding='SAME')
def blur2(x2):
    kernel_var = gauss_kernel(21, 3, 1)
    return tf.nn.depthwise_conv2d(x2 , kernel_var, [1, 1, 1, 1], padding='SAME')
def sobel(x):
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 3])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])
    # Shape = height x width.
    #image = tf.placeholder(tf.float32, shape=[None, None]) 
    # Shape = 1 x height x width x 1.
    #image_resized = tf.expand_dims(tf.expand_dims(image, 0), 3)
 
    filtered_x = tf.nn.conv2d(x, sobel_x_filter,
                          strides=[1, 1, 1, 1], padding='SAME')
    filtered_y = tf.nn.conv2d(x, sobel_y_filter,
                          strides=[1, 1, 1, 1], padding='SAME')
    return tf.add(filtered_x,filtered_y)
def gradient(x, direction):
    smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
    smooth_kernel_y = tf.transpose(smooth_kernel_x, [1, 0, 2, 3])
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    return tf.abs(tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME'))#154
def tv_grad(x):
    input_R = tf.image.rgb_to_grayscale(x)
    return gradient(input_R,"x")+gradient(input_R,"y")
