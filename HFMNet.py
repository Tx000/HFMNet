import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn
import utils_best
def _instance_norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
    return scale * normalized + shift
def lrelu(x):
    return tf.maximum(x*0.2,x)
def upsample(x1,x2,output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )
    #deconv_output =  tf.concat([deconv, x2],3)
    deconv_output =  deconv
    #deconv_output = tf.reshape(deconv,deconv.get_shape())
    deconv_output.set_shape([None, None, None, output_channels])
    return deconv_output
# unet_backbone
def unet_backbone(input_image):
    with tf.variable_scope('unet_backbone', reuse=tf.AUTO_REUSE):
        input_smooth=utils_best.blur(input_image)
        input_edge=utils_best.tv_grad(input_image)
        input_three1=tf.concat([input_image,input_smooth],-1)
        input_three=tf.concat([input_three1,input_edge],-1)
        print("############edge shape####################")
        print(input_edge.get_shape())
        print("############edge shape####################")
        input1 = slim.conv2d(input_three, 64, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='input_conv1')#50*50


        conv1_1=slim.conv2d(input1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        conv1_2=slim.conv2d(conv1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')

        conv1_3=_instance_norm(conv1_2)

        conv1_4=tf.add(conv1_3,input1)
        conv2_1=slim.conv2d(conv1_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_3')
        conv2_1_1=tf.add(conv2_1,conv1_1)
        conv2_2=slim.conv2d(conv2_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_4')
        conv2_3=_instance_norm(conv2_2)

        conv2_4=tf.add(conv2_3,conv1_4)

        pool1=slim.conv2d(conv2_4,128,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling1' )#25*25
        conv4_1=slim.conv2d(pool1,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        conv4_2=slim.conv2d(conv4_1,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
        conv4_3=_instance_norm(conv4_2)

        conv4_4=tf.add(conv4_3,pool1)


        up1 =  upsample(conv4_4,input1,64,128) #50*50
        up1_1 = tf.concat([up1, conv1_4],3)
        conv7_1=slim.conv2d(up1_1, 64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7_2=slim.conv2d(conv7_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')
        conv7_3=_instance_norm(conv7_2)

        conv7_4=tf.add(conv7_3,up1)
        conv8_1=slim.conv2d(conv7_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_3')

        conv8_1_1=tf.add(conv7_1,conv8_1)
        conv8_2=slim.conv2d(conv8_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_4')
        conv8_3=_instance_norm(conv8_2)

        conv8_4=tf.add(conv8_3,conv7_4)


        deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 3, 64], stddev=0.02))
        #conv10 = tf.nn.conv2d_transpose(conv9_4, deconv_filter, tf.shape(input), strides=[1, 2, 2, 1])
        conv10 = tf.nn.conv2d_transpose(conv8_4, deconv_filter, output_shape=[tf.shape(input_image)[0],tf.shape(input_image)[1],tf.shape(input_image)[2],3], strides=[1, 2, 2, 1])

        out = slim.conv2d(conv10, 3, [3, 3],rate=1,activation_fn=nn.tanh,scope='out') * 0.58 + 0.52

    return out

def unet_backbone_add(input_image):
    with tf.variable_scope('unet_backbone', reuse=tf.AUTO_REUSE):
        input_smooth=utils_best.blur(input_image)
        input_edge=utils_best.tv_grad(input_image)
        input_three1=tf.concat([input_image,input_smooth],-1)
        input_three=tf.concat([input_three1,input_edge],-1)
        print("############edge shape####################")
        print(input_edge.get_shape())
        print("############edge shape####################")
        input1 = slim.conv2d(input_three, 64, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='input_conv1')#50*50


        conv1_1=slim.conv2d(input1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        conv1_2=slim.conv2d(conv1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
        conv1_3=_instance_norm(conv1_2)

        conv1_4=tf.add(conv1_3,input1)

        conv2_1=slim.conv2d(conv1_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_3')
        conv2_1_1=tf.add(conv2_1,conv1_1)
        conv2_2=slim.conv2d(conv2_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_4')
        conv2_3=_instance_norm(conv2_2)

        conv2_4=tf.add(conv2_3,conv1_4)

        conv3_1=slim.conv2d(conv2_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
        conv3_1_1=tf.add(conv3_1,conv2_1)
        conv3_2=slim.conv2d(conv3_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_2')
        conv3_3=_instance_norm(conv3_2)

        conv3_4=tf.add(conv3_3,conv2_4)

        conv4_1=slim.conv2d(conv3_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_1')
        conv4_1_1=tf.add(conv4_1,conv3_1)
        conv4_2=slim.conv2d(conv4_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_2')
        conv4_3=_instance_norm(conv4_2)

        conv4_4=tf.add(conv4_3,conv3_4)

        conv5_1=slim.conv2d(conv4_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_1')
        conv5_1_1=tf.add(conv5_1,conv4_1)
        conv5_2=slim.conv2d(conv5_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_2')
        conv5_3=_instance_norm(conv5_2)

        conv5_4=tf.add(conv5_3,conv4_4)

        pool1=slim.conv2d(conv5_4,64,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling1' )#25*25

        conv6_1=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
        conv6_2=slim.conv2d(conv6_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')
        conv6_3=_instance_norm(conv6_2)

        conv6_4=tf.add(conv6_3,pool1)

        conv7_1=slim.conv2d(conv6_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7_2=slim.conv2d(conv7_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')
        conv7_3=_instance_norm(conv7_2)

        conv7_4=tf.add(conv7_3,conv7_1)

        up1 =  upsample(conv7_4,input1,64,64) #50*50
        #up1_1 = tf.concat([up1, conv1_4],3)
        conv8_1=slim.conv2d(up1, 64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
        conv8_2=slim.conv2d(conv8_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')
        conv8_3=_instance_norm(conv8_2)

        conv8_4=tf.add(conv8_3,up1)

        conv9_1=slim.conv2d(conv8_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
        conv9_1_1=tf.add(conv9_1,conv8_1)
        conv9_2=slim.conv2d(conv9_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')
        conv9_3=_instance_norm(conv9_2)

        conv9_4=tf.add(conv9_3,conv8_4)

        conv10_1=slim.conv2d(conv9_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv10_1')
        conv10_1_1=tf.add(conv10_1,conv9_1)
        conv10_2=slim.conv2d(conv10_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv10_2')
        conv10_3=_instance_norm(conv10_2)

        conv10_4=tf.add(conv10_3,conv9_4)

        conv11_1=slim.conv2d(conv10_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv11_1')
        conv11_1_1=tf.add(conv11_1,conv9_1)
        conv11_2=slim.conv2d(conv11_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv11_2')
        conv11_3=_instance_norm(conv11_2)

        conv11_4=tf.add(conv11_3,conv10_4)

        conv12_1=slim.conv2d(conv11_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv12_1')
        conv12_1_1=tf.add(conv12_1,conv11_1)
        conv12_2=slim.conv2d(conv12_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv12_2')
        conv12_3=_instance_norm(conv12_2)

        conv12_4=tf.add(conv12_3,conv11_4)

        deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 3, 64], stddev=0.02))
        #conv10 = tf.nn.conv2d_transpose(conv9_4, deconv_filter, tf.shape(input), strides=[1, 2, 2, 1])
        conv13 = tf.nn.conv2d_transpose(conv12_4, deconv_filter, output_shape=[tf.shape(input_image)[0],tf.shape(input_image)[1],tf.shape(input_image)[2],3], strides=[1, 2, 2, 1])

        out = slim.conv2d(conv13, 3, [3, 3],rate=1,activation_fn=nn.tanh,scope='out') * 0.58 + 0.52

    return out

def unet_backbone_double(input_image):
    with tf.variable_scope('unet_backbone', reuse=tf.AUTO_REUSE):
        input_smooth=utils_best.blur(input_image)
        input_edge=utils_best.tv_grad(input_image)
        input_three1=tf.concat([input_image,input_smooth],-1)
        input_three=tf.concat([input_three1,input_edge],-1)
        print("############edge shape####################")
        print(input_edge.get_shape())
        print("############edge shape####################")

        input1 = slim.conv2d(input_three, 64, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='input_conv1')#50*50

        conv1_1=slim.conv2d(input1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        conv1_2=slim.conv2d(conv1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
        conv1_3=_instance_norm(conv1_2)

        conv1_4=tf.add(conv1_3,input1)

        conv2_1=slim.conv2d(conv1_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_3')
        conv2_1_1=tf.add(conv2_1,conv1_1)
        conv2_2=slim.conv2d(conv2_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_4')
        conv2_3=_instance_norm(conv2_2)

        conv2_4=tf.add(conv2_3,conv1_4)

        pool1=slim.conv2d(conv2_4,128,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling1' )#25*25
        conv4_1=slim.conv2d(pool1,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        conv4_2=slim.conv2d(conv4_1,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
        conv4_3=_instance_norm(conv4_2)

        conv4_4=tf.add(conv4_3,pool1)

        up1 =  upsample(conv4_4,input1,64,128) #50*50
        #up1_1 = tf.concat([up1, conv1_4],3)
        conv7_1=slim.conv2d(up1, 64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7_2=slim.conv2d(conv7_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')
        conv7_3=_instance_norm(conv7_2)

        conv7_4=tf.add(conv7_3,up1)

        conv8_1=slim.conv2d(conv7_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_3')
        conv8_1_1=tf.add(conv7_1,conv8_1)
        conv8_2=slim.conv2d(conv8_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_4')
        conv8_3=_instance_norm(conv8_2)

        conv8_4=tf.add(conv8_3,conv7_4)


        deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 3, 64], stddev=0.02))
        #conv10 = tf.nn.conv2d_transpose(conv9_4, deconv_filter, tf.shape(input), strides=[1, 2, 2, 1])
        conv10 = tf.nn.conv2d_transpose(conv8_4, deconv_filter, output_shape=[tf.shape(input_image)[0],tf.shape(input_image)[1],tf.shape(input_image)[2],3], strides=[1, 2, 2, 1])

        out = slim.conv2d(conv10, 3, [3, 3],rate=1,activation_fn=nn.tanh,scope='out') * 0.58 + 0.52

    return out


def net_three_fusion(input_image):
    with tf.variable_scope('net_three_fusion', reuse=tf.AUTO_REUSE):
        input_smooth=utils_best.blur(input_image)
        input_edge=utils_best.tv_grad(input_image)
        input_three1=tf.concat([input_image,input_smooth],-1)
        input_three=tf.concat([input_three1,input_edge],-1)
        print("############edge shape####################")
        print(input_edge.get_shape())
        print("############edge shape####################")
        input1 = slim.conv2d(input_three, 64, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='input_conv1')#50*50
#pixelshuffle?
        #input_s = slim.conv2d(input_smooth, 3, [3, 3], rate=1, activation_fn=None, scope='input_conv2')
        #input_s = tf.nn.sigmoid(input_s)# 0-1
# upsample output=3
        input_c = slim.conv2d(input1, 3, [3, 3], rate=1, activation_fn=None, scope='input_c')
        mask_color= upsample(input_c,input_smooth,3,3)
        mask_color= tf.nn.sigmoid(mask_color)# 0-1
        mask_color_SR = upsample(input_c,input_smooth,3,3)## for training
        out_color = mask_color_SR*mask_color

        conv1_1=slim.conv2d(input1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        conv1_2=slim.conv2d(conv1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
        conv1_3=_instance_norm(conv1_2)

        conv1_4=tf.add(conv1_3,input1)
        conv2_1=slim.conv2d(conv1_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_3')
        conv2_1_1=tf.add(conv2_1,conv1_1)
        conv2_2=slim.conv2d(conv2_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_4')
        conv2_3=_instance_norm(conv2_2)

        conv2_4=tf.add(conv2_3,conv1_4)

        pool1=slim.conv2d(conv2_4,128,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling1' )#25*25
        conv4_1=slim.conv2d(pool1,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        conv4_2=slim.conv2d(conv4_1,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
        conv4_3=_instance_norm(conv4_2)

        conv4_4=tf.add(conv4_3,pool1)


        up1 =  upsample(conv4_4,input1,64,128) #50*50
        up1_1 = tf.concat([up1, conv1_4],3)
        conv7_1=slim.conv2d(up1_1, 64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7_2=slim.conv2d(conv7_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')
        conv7_3=_instance_norm(conv7_2)

        conv7_4=tf.add(conv7_3,up1)
        conv8_1=slim.conv2d(conv7_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_3')

        conv8_1_1=tf.add(conv7_1,conv8_1)
        conv8_2=slim.conv2d(conv8_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_4')
        conv8_3=_instance_norm(conv8_2)

        conv8_4=tf.add(conv8_3,conv7_4)
### high-level features
        input_e = slim.conv2d(input_edge, 1, [3, 3],  rate=1, activation_fn=None, scope='input_conv3')
        input_e = tf.nn.sigmoid(input_e)# 0-1
# upsample output=3
        input_e2= slim.conv2d(conv8_4, 1, [3, 3],  rate=1, activation_fn=None, scope='input_e')
        mask_edge= upsample(input_e2,input_edge,1,1)
        mask_edge_SR = upsample(input_e2,input_edge,1,1)
        mask_edge= tf.nn.sigmoid(mask_edge)# 0-1 
        out_edge = mask_edge_SR*mask_edge

        deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 3, 64], stddev=0.02))
        #conv10 = tf.nn.conv2d_transpose(conv9_4, deconv_filter, tf.shape(input), strides=[1, 2, 2, 1])
        conv10 = tf.nn.conv2d_transpose(conv8_4, deconv_filter, output_shape=[tf.shape(input_image)[0],tf.shape(input_image)[1],tf.shape(input_image)[2],3], strides=[1, 2, 2, 1])

        conv11 = tf.concat([out_color,out_edge],-1)
        conv12 =tf.concat([conv11,conv10],-1)
        out = slim.conv2d(conv12, 3, [3, 3],rate=1,activation_fn=nn.tanh,scope='out') * 0.58 + 0.52

    return out,mask_color_SR,mask_edge_SR,mask_color, mask_edge

def net_three_double(input_image):
    with tf.variable_scope('net_three_double', reuse=tf.AUTO_REUSE):
        input_smooth=utils_best.blur(input_image)
        input_edge=utils_best.tv_grad(input_image)
        input_three1=tf.concat([input_image,input_smooth],-1)
        input_three=tf.concat([input_three1,input_edge],-1)
        print("############edge shape####################")
        print(input_edge.get_shape())
        print("############edge shape####################")
        input1 = slim.conv2d(input_three, 64, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='input_conv1')#50*50
#pixelshuffle?
        #input_s = slim.conv2d(input_smooth, 3, [3, 3], rate=1, activation_fn=None, scope='input_conv2')
        #input_s = tf.nn.sigmoid(input_s)# 0-1
# upsample output=3
        input_c = slim.conv2d(input1, 3, [3, 3], rate=1, activation_fn=None, scope='input_c')
        mask_color= upsample(input_c,input_smooth,3,3)
        mask_color= tf.nn.sigmoid(mask_color)# 0-1
        mask_color_SR = upsample(input_c,input_smooth,3,3)## for training
        out_color = mask_color_SR*mask_color

        input_e= slim.conv2d(input1, 1, [3, 3],  rate=1, activation_fn=None, scope='input_e')
        mask_edge= upsample(input_e,input_edge,1,1)
        mask_edge_SR = upsample(input_e,input_edge,1,1)
        mask_edge= tf.nn.sigmoid(mask_edge)# 0-1 
        out_edge = mask_edge_SR*mask_edge

        conv1_1=slim.conv2d(input1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        conv1_2=slim.conv2d(conv1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
        conv1_3=_instance_norm(conv1_2)

        conv1_4=tf.add(conv1_3,input1)
        conv2_1=slim.conv2d(conv1_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_3')
        conv2_1_1=tf.add(conv2_1,conv1_1)
        conv2_2=slim.conv2d(conv2_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_4')
        conv2_3=_instance_norm(conv2_2)

        conv2_4=tf.add(conv2_3,conv1_4)

        pool1=slim.conv2d(conv2_4,128,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling1' )#25*25
        conv4_1=slim.conv2d(pool1,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        conv4_2=slim.conv2d(conv4_1,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
        conv4_3=_instance_norm(conv4_2)

        conv4_4=tf.add(conv4_3,pool1)


        up1 =  upsample(conv4_4,input1,64,128) #50*50

        up1_1 = tf.concat([up1, conv1_4],3)
        conv7_1=slim.conv2d(up1_1, 64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7_2=slim.conv2d(conv7_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')
        conv7_3=_instance_norm(conv7_2)

        conv7_4=tf.add(conv7_3,up1)
        conv8_1=slim.conv2d(conv7_4,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_3')

        conv8_1_1=tf.add(conv7_1,conv8_1)
        conv8_2=slim.conv2d(conv8_1_1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_4')
        conv8_3=_instance_norm(conv8_2)

        conv8_4=tf.add(conv8_3,conv7_4)
### high-level features
# upsample output=3
        input_c2 = slim.conv2d(conv8_4, 3, [3, 3], rate=1, activation_fn=None, scope='input_c2')
        mask_color2= upsample(input_c2,input_smooth,3,3)
        mask_color2= tf.nn.sigmoid(mask_color2)# 0-1
        mask_color_SR2 = upsample(input_c2,input_smooth,3,3)## for training
        out_color2 = mask_color_SR2*mask_color2

        input_e2= slim.conv2d(conv8_4, 1, [3, 3],  rate=1, activation_fn=None, scope='input_e2')
        mask_edge2= upsample(input_e2,input_edge,1,1)
        mask_edge_SR2 = upsample(input_e2,input_edge,1,1)
        mask_edge2= tf.nn.sigmoid(mask_edge2)# 0-1 
        out_edge2 = mask_edge_SR2*mask_edge2

        deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 3, 64], stddev=0.02))
        #conv10 = tf.nn.conv2d_transpose(conv9_4, deconv_filter, tf.shape(input), strides=[1, 2, 2, 1])
        conv10 = tf.nn.conv2d_transpose(conv8_4, deconv_filter, output_shape=[tf.shape(input_image)[0],tf.shape(input_image)[1],tf.shape(input_image)[2],3], strides=[1, 2, 2, 1])

        conv11 = tf.concat([out_color,out_edge],-1)
        conv12 =tf.concat([out_color2,out_edge2],-1) 
        conv13 = tf.concat([conv11,conv12],-1)
        conv14 =tf.concat([conv13,conv10],-1)
        out = slim.conv2d(conv14, 3, [3, 3],rate=1,activation_fn=nn.tanh,scope='out') * 0.58 + 0.52

    return out,mask_color_SR,mask_edge_SR,mask_color_SR2,mask_edge_SR2


def adversarial1(image_):

    with tf.variable_scope("discriminator1"):

        conv1 = _conv_layer(image_, 48, 11, 4, batch_nn = False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 128, 3, 2)#下采样了，stride=2
        
        flat_size = 128 * 7 * 7
        conv5_flat = tf.reshape(conv5, [-1, flat_size])

        W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

        fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

        W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)
    
    return adv_out
def adversarial2(image_):
    with tf.variable_scope("discriminator2"):

        conv1 = _conv_layer(image_, 64, 9, 4, batch_nn = False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 256, 3, 2)#下采样了，stride=2
        conv6 = _conv_layer(conv5, 256, 3, 1)#下采样了，stride=2        
        flat_size = 256 * 8 * 8
        conv5_flat = tf.reshape(conv5, [-1, flat_size])

        W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

        fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

        W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)
    
    return adv_out
def weight_variable(shape, name):

    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):

    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def leaky_relu(x, alpha = 0.2):
    return tf.maximum(alpha * x, x)

def _conv_layer(net, num_filters, filter_size, strides, batch_nn=True):
    
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME') + bias   
    net = leaky_relu(net)

    if batch_nn:
        net = _instance_norm(net)

    return net

def _conv_init_vars(net, out_channels, filter_size, transpose=False):

    _, rows, cols, in_channels = [i.value for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init
