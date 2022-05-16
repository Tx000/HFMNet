from __future__ import print_function

import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
import HFMNet
#import utils_best
from utils import *
from utils3 import *
PATCH_WIDTH = 128
PATCH_HEIGHT = 128
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

def concat(layers):
    return tf.concat(layers, axis=3)

def adversarial_Net(input_im2):
    with tf.variable_scope('adversarial', reuse=tf.AUTO_REUSE):
        out = HFMNet.adversarial2(input_im2)
    return out
class lowlight_enhance(object):
    def __init__(self, sess):
        self.sess = sess
        # build the model
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')
        self.input_mos = tf.placeholder(tf.float32, [None, None, None, 3], name='input_mos')
        self.adv_ = tf.placeholder(tf.float32, [None, 1])

        output,out_color,out_edge,mask_color,mask_edge = HFMNet.net_three_fusion(self.input_low)#输入低光图像
        self.output_S = output
        self.output_c = out_color
        self.output_e = out_edge
        self.mask_c = mask_color
        self.mask_e = mask_edge

        enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(output), [-1, PATCH_WIDTH * PATCH_HEIGHT])
        #high_gray = tf.reshape(tf.image.rgb_to_grayscale(self.input_high),[-1, PATCH_WIDTH * PATCH_HEIGHT])
        mos_gray = tf.reshape(tf.image.rgb_to_grayscale(self.input_mos),[-1, PATCH_WIDTH * PATCH_HEIGHT])
        # push randomly the enhanced or dslr image to an adversarial CNN-discriminator
        adversarial_ = tf.multiply(enhanced_gray, 1 - self.adv_) + tf.multiply(mos_gray, self.adv_)
        adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])
        discrim_predictions = adversarial_Net(adversarial_image)
        # loss
        # 1) adversarial loss
        discrim_target = tf.concat([self.adv_, 1 - self.adv_], 1)
        self.loss_discrim = -tf.reduce_sum(discrim_target * tf.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))
        self.loss_texture = -self.loss_discrim

        correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
        discim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        #1-1) color mask loss
        self.loss_color = tf.reduce_mean(tf.abs(self.output_c- self.input_high))+0.2*self.mutual_i_loss(self.output_c,self.input_high)+0.15*self.mutual_i_input_loss(self.output_c,self.input_high)
        #1-2) edge mask loss
        #self.loss_edge = tf.reduce_mean(tf.abs(self.tv_grad(self.input_high)-self.tv_grad(self.output_e)))
        self.loss_edge = tf.reduce_mean(tf.square(self.tv_grad2(self.output_e) -self.tv_grad(self.input_high)))
        #2) MSE loss
        #self.recon_loss_high = tf.reduce_mean(tf.abs(self.output_S - self.input_high))
        self.loss_ssim = 1-tf.reduce_mean(tf.image.ssim(self.output_S * 255, self.input_high* 255, max_val=255))
        #self.loss_color = tf.reduce_mean(tf.abs(self.color_s(output) - self.color_s(self.input_high)))
        self.recon_local = tf.reduce_mean(tf.square(self.output_S - self.input_high))
        #self.loss_tv = tf.reduce_mean(tf.square(self.tv_grad(self.output_S) -self.tv_grad(self.input_high)))
        #self.loss_final = 0.32*self.recon_loss_high+self.loss_ssim+0.01*self.loss_color+0.68*self.recon_local+0.01*self.loss_tv
        #self.loss_final = 0.68*self.recon_loss_high+self.loss_ssim+0.01*self.loss_color+0.32*self.recon_local+0.01*self.loss_tv
        #self.loss_final = self.recon_local+0.1*self.loss_texture
        self.loss_final = self.loss_ssim+0.005*self.loss_texture+0.001*self.loss_color+self.loss_edge

        #metric
        self.psnr_R = tf.reduce_mean(tf.image.psnr(self.output_S * 255, self.input_high* 255, max_val=255))
        self.ssim_R = tf.reduce_mean(tf.image.ssim(self.output_S * 255, self.input_high* 255, max_val=255))


        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.var_Decom = [var for var in tf.trainable_variables() if 'net_three_fusion' in var.name]
        self.var_discrim = [var for var in tf.trainable_variables() if 'adversarial' in var.name]
        self.train_op_Decom = optimizer.minimize(self.loss_final, var_list = self.var_Decom)
        self.train_op_discrim = optimizer.minimize(self.loss_discrim, var_list = self.var_discrim)


        self.sess.run(tf.global_variables_initializer())
        self.saver_Decom = tf.train.Saver(var_list = self.var_Decom, max_to_keep=0)
        print("[*] Initialize model successfully...")


    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3])

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))#154
    def tv_grad(self,input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        return self.gradient(input_R,"x")+self.gradient(input_R,"y")
    def tv_grad2(self,input_I):
        return self.gradient(input_I,"x")+self.gradient(input_I,"y")


    def smooth(self, input_I, input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        return tf.reduce_mean(self.gradient(input_I, "x") * tf.exp(-10 * self.ave_gradient(input_R, "x")) + self.gradient(input_I, "y") * tf.exp(-10 * self.ave_gradient(input_R, "y")))

    def color_s(self, input_c): 
        max_channel = tf.reduce_max(input_c,axis=-1,keep_dims=True)
        min_channel = tf.reduce_min(input_c,axis=-1,keep_dims=True)
        res_channel = (max_channel- min_channel)/(max_channel+0.01)
        return res_channel
    def mutual_i_loss(self, input_I_low, input_I_high):
        input_gray = tf.image.rgb_to_grayscale(input_I_low)
        input_gray2 = tf.image.rgb_to_grayscale(input_I_high)
        low_gradient_x = gradient(input_gray, "x")
        high_gradient_x = gradient(input_gray2, "x")
        x_loss = (low_gradient_x + high_gradient_x)* tf.exp(-10*(low_gradient_x+high_gradient_x))
        low_gradient_y = gradient(input_gray, "y")
        high_gradient_y = gradient(input_gray2, "y")
        y_loss = (low_gradient_y + high_gradient_y) * tf.exp(-10*(low_gradient_y+high_gradient_y))
        mutual_loss = tf.reduce_mean( x_loss + y_loss) 
        return mutual_loss

    def mutual_i_input_loss(self, input_I_low, input_im):
        input_gray = tf.image.rgb_to_grayscale(input_im)
        input_gray2 = tf.image.rgb_to_grayscale(input_I_low)
        low_gradient_x = gradient(input_gray2, "x")
        input_gradient_x = gradient(input_gray, "x")
        x_loss = tf.abs(tf.div(low_gradient_x, tf.maximum(input_gradient_x, 0.01)))
        low_gradient_y = gradient(input_gray2, "y")
        input_gradient_y = gradient(input_gray, "y")
        y_loss = tf.abs(tf.div(low_gradient_y, tf.maximum(input_gradient_y, 0.01)))
        mut_loss = tf.reduce_mean(x_loss + y_loss) 
        return mut_loss

    def evaluate(self, epoch_num, eval_low_data, eval_high_data, eval_mos_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))
        psnr1=0.0
        ssim1=0.0
        avg_psnr=0.0
        avg_ssim=0.0
        psnr_R=self.psnr_R
        ssim_R=self.ssim_R

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)
            input_high_eval = np.expand_dims(eval_high_data[idx], axis=0)
            input_mos_eval = np.expand_dims(eval_mos_data[idx], axis=0)
            if train_phase == "Decom_CEM":
                psnr,ssim= self.sess.run([psnr_R,ssim_R],feed_dict={self.input_low: input_low_eval,self.input_high: input_high_eval})
                psnr1+=psnr
                ssim1+=ssim
                if epoch_num == 1000:
                    result_1= self.sess.run(self.output_S, feed_dict={self.input_low: input_low_eval,self.input_mos: input_mos_eval})
                    save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1)
                    result_c= self.sess.run(self.mask_c, feed_dict={self.input_low: input_low_eval,self.input_mos: input_mos_eval})
                    save_images(os.path.join(sample_dir, 'color_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_c)
                    result_e= self.sess.run(self.mask_e, feed_dict={self.input_low: input_low_eval,self.input_mos: input_mos_eval})
                    save_images(os.path.join(sample_dir, 'edge_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_e)
        avg_psnr=psnr1/(len(eval_low_data))
        avg_ssim=ssim1/(len(eval_low_data))
        print("psnr: %.4f,ssim: %.4f" \
                   % (avg_psnr,avg_ssim))
        f = open("psnr_CEM_ssim.txt", "a+")
        #f = open("psnr_semi_0.1.txt", "a+")
        print("%d---PSNR %.4f , SSIM  %.4f ---" % (epoch_num,avg_psnr,avg_ssim), file=f)
        f.write('\n')
        f.close()

    def train(self, train_low_data, train_high_data, train_mos_data, eval_low_data, eval_high_data, eval_mos_data,batch_size, patch_size, epoch, lr, sample_dir, ckpt_dir, eval_every_epoch, train_phase):
        assert len(train_low_data) == len(train_high_data)
        print("total train number")
        print(len(train_low_data))
        numBatch = len(train_low_data) // int(batch_size)

        # load pretrained model
        if train_phase == "Decom_CEM":
            train_op = self.train_op_Decom
            train_op_semi = self.train_op_discrim
            train_loss = self.loss_final
            train_loss_ssim = self.loss_ssim
            train_loss_color = self.loss_color
            train_loss_edge = self.loss_edge
            train_loss_discrim = self.loss_discrim
            saver = self.saver_Decom
        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_mos = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(train_low_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(train_high_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_mos[patch_id, :, :, :] = data_augmentation(train_mos_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)                    
                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data  = zip(*tmp)

                # train
                _,loss,loss_ssim,loss_color,loss_edge  = self.sess.run([train_op,train_loss,train_loss_ssim,train_loss_color,train_loss_edge], feed_dict={self.input_low: batch_input_low, \
                                                                           self.input_high: batch_input_high, \
                                                                           self.input_mos: batch_input_mos, \
                                                                           self.lr: lr[epoch], self.adv_: swaps})
                _, loss_semi = self.sess.run([train_op_semi, train_loss_discrim], feed_dict={self.input_low: batch_input_low, \
                                                                           self.input_high: batch_input_high, \
                                                                           self.input_mos: batch_input_mos, \
                                                                           self.lr: lr[epoch],self.adv_: swaps})

                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.4f, loss_ssim: %.4f,loss_color: %.4f,loss_edge: %.4f,loss_semi: %.4f" \
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time,loss,loss_ssim,loss_color,loss_edge,loss_semi))
                iter_num += 1

            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                self.evaluate(epoch + 1, eval_low_data,eval_high_data, eval_mos_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(saver, iter_num, ckpt_dir, "CEM-%s" % train_phase)

        print("[*] Finish training for phase %s." % train_phase)


    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, test_high_data_names, save_dir, decom_flag):
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.saver_Decom, './22.64_0.8589')
        
        print("[*] Testing...")
        print(len(test_low_data))
        print(len(test_high_data))
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]
            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            print(test_high_data_names[idx])

            input_high_test = np.expand_dims(test_high_data[idx], axis=0)
            #[R_low, I_low] = self.sess.run([self.output_R_low, self.output_I_low], feed_dict = {self.input_low: input_low_test})
            output=self.sess.run(self.output_S,feed_dict = {self.input_low: input_low_test,self.input_high: input_high_test})
            save_images(os.path.join(save_dir, name + "." + suffix), output)

    def test_low(self, test_low_data,test_low_data_names, save_dir, decom_flag):
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        #load_model_status_Decom, _ = self.load(self.saver_Decom, './save_model/22.64_0.8589') 
        load_model_status_Decom, _ = self.load(self.saver_Decom, './22.64_0.8589') 
        #load_model_status_Decom, _ = self.load(self.saver_Decom, './save_model/22.76_0.8680') 
        #load_model_status_Decom, _ = self.load(self.saver_Decom, './save_model/22.99')          
        print("[*] Testing...")
        print(len(test_low_data))
        total_run_time = 0.0        
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            #print(test_high_data_names[idx])
            start_time = time.time()
            output=self.sess.run(self.output_S,feed_dict = {self.input_low: input_low_test})
            total_run_time += time.time() - start_time
            save_images(os.path.join(save_dir, name + "." + suffix), output)
        #ave_run_time = total_run_time / (float(len(test_low_data))-1)
        ave_run_time = total_run_time / 100
        print("[*] Average run time: %.4f" % ave_run_time)

