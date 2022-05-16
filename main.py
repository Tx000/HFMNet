# python main.py
from __future__ import print_function
import os
import argparse
import random
from glob import glob
import random
from PIL import Image
import tensorflow as tf

from model import lowlight_enhance
from utils import *


parser = argparse.ArgumentParser(description='')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.5, help="0 to 1, gpu memory usage")
parser.add_argument('--phase', dest='phase', default='test', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=12000, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=30, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=128, help='patch size')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=50, help='evaluating and saving checkpoints every #  epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint_TIM/checkpoint_ssim', help='directory for checkpoints')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample_CEM', help='directory for evaluating outputs')

parser.add_argument('--save_dir', dest='save_dir', default='./test_results/LOL', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./test_data', help='directory for testing inputs')
#parser.add_argument('--test_dir', dest='test_dir', default='./test_data', help='directory for testing inputs')
parser.add_argument('--decom', dest='decom', default=0, help='decom flag, 0 for enhanced results only and 1 for decomposition results')

args = parser.parse_args()

def lowlight_train(lowlight_enhance):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    lr = args.start_lr * np.ones([args.epoch])
    lr[20:] = lr[0] / 10.0

    train_low_data = []
    train_high_data = []
    train_mos_data = []
    train_low_data_names = glob('./train_data/Our_low/*.png') 
    train_low_data_names.sort()
    train_high_data_names = glob('./train_data/Our_normal/*.png') 
    train_high_data_names.sort()
    train_mos_data_names = glob('./train_data/Our_normal/*.png') 
    #random.shuffle(train_mos_data_names)
    #train_mos_data_names = random.shuffle(train_mos_data_names)
    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))

    for idx in range(len(train_low_data_names)):
        print(train_low_data_names[idx])
        print(train_high_data_names[idx])
        print(train_mos_data_names[idx])
        low_im = load_images(train_low_data_names[idx])
        train_low_data.append(low_im)
        high_im = load_images(train_high_data_names[idx])
        train_high_data.append(high_im)
        mos_im = load_images(train_mos_data_names[idx])
        train_mos_data.append(mos_im)

    eval_low_data = []
    eval_high_data = []
    eval_mos_data = []

    eval_low_data_name = glob('test_data/Our_low_test/*.*')
    eval_high_data_name = glob('test_data/Our_normal_test/*.*')
    eval_mos_data_name = glob('test_data/Our_normal_test/*.*')
    eval_low_data_name.sort()
    eval_high_data_name.sort()

    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_images(eval_low_data_name[idx])
        eval_low_data.append(eval_low_im)
        eval_high_im = load_images(eval_high_data_name[idx])
        eval_high_data.append(eval_high_im)
        eval_mos_im = load_images(eval_mos_data_name[idx])
        eval_mos_data.append(eval_mos_im)


    lowlight_enhance.train(train_low_data, train_high_data, train_mos_data, eval_low_data, eval_high_data, eval_mos_data,batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Decom_CEM'), eval_every_epoch=args.eval_every_epoch, train_phase="Decom_CEM")

    #lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, eval_high_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Restor'), eval_every_epoch=args.eval_every_epoch, train_phase="Restor")

    #lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, eval_high_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Adjust'), eval_every_epoch=args.eval_every_epoch, train_phase="Adjust")

    #lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, lr=lr, #sample_dir=args.sample_dir, ckpt_dir=os.path.join(args.ckpt_dir, 'Relight'), eval_every_epoch=args.eval_every_epoch, train_phase="Relight")


def lowlight_test(lowlight_enhance):
    if args.test_dir == None:
        print("[!] please provide --test_dir")
        exit(0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    #test_low_data_name = glob(os.path.join(args.test_dir) +'/Our_low_test'+ '/*.*')
    test_low_data_name = glob(os.path.join(args.test_dir) +'/Our_low_test'+ '/*.*')
    #test_low_data_name = glob(os.path.join(args.test_dir)  +'/resize12.21_2'+ '/*.*')
    test_high_data_name = glob(os.path.join(args.test_dir) +'/Our_normal_test'+ '/*.*') 
    test_low_data = []
    test_high_data = []
    for idx in range(len(test_low_data_name)):
        test_low_im = load_images(test_low_data_name[idx])
        test_high_im = load_images(test_high_data_name[idx])
        #print(test_low_data_name[idx])
        #print("shape\n")
        #print(test_low_im.shape)
        test_low_data.append(test_low_im)
        #test_high_im = load_images(test_high_data_name[idx])
        test_high_data.append(test_high_im)

    #lowlight_enhance.test(test_low_data, test_high_data, test_low_data_name,test_high_data_name, save_dir=args.save_dir, decom_flag=args.decom)
    lowlight_enhance.test_low(test_low_data, test_low_data_name, save_dir=args.save_dir, decom_flag=args.decom)

def main(_):
    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        with tf.Session(config=config) as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] CPU\n")
        with tf.Session() as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)

if __name__ == '__main__':
    tf.app.run()
