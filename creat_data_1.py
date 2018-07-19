#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'ccchen_706@126.com'
'''
import os
from gconfig import *
from xml2csv import creat_csv
from gen_tfrecord import crate_tfrecord
from creat_pbtxt import write_config_pbtxt
from config_net import creat_train_config_file

train_param_str=' --train_dir='+os.path.abspath(output_train_dir).replace('\\','/')+'/'
config_param_str=' --pipeline_config_path='+os.path.abspath(output_train_dir).replace('\\','/')+'/'+output_train_dir+'.config'

if __name__ == '__main__':
    creat_csv()
    crate_tfrecord()
    write_config_pbtxt()
    creat_train_config_file()
    print('all prepare is alread!')
    print('please use command to train net:\n')
    print('python train.py --logtostderr'+train_param_str+config_param_str)
    print('please use command to look tensorboard:\n')
    print('tensorboard --logdir='+os.path.abspath(output_train_dir).replace('\\','/')+'/')





