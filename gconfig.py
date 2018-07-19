#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'ccchen_706@126.com'
'''

#---------------------------------------------------
#1 train data prepare config
# Train_Data_Path='train_data'
# Test_Data_Path='test_data'
# Train_File_Name='train_bullet_hole_v1'
# Test_File_Name='test_bullet_hole_v1'
#
# #1 train config net
# Class_Dic={'bullet_hole':1,'chest_bitmap':2}
# Config_path='Config_template'
# Config_template_file='ssdlite_mobilenet_v1_coco.config'
# image_w=640
# image_h=480
# train_num_step=200000
# train_batch_size=6
#
# #2 train data out config
# output_train_dir='bullet_hole_output'
#
# #3 model out config
# module_out_put_dir='model_'+output_train_dir
# model_ckpt=33426
#
# #4 run test config
# use_camer=False
# run_test_file_dir='run_test_data'
# run_test_file_out_dir=run_test_file_dir+'/'+'test_data_out/'
# imput_image_w=1280
# imput_image_h=960

#---------------------------------------------------

#1 train data prepare config
# Train_Data_Path=['train_data_1/1','train_data_1/2']
# Test_Data_Path=['test_data_1/1','test_data_1/2','test_data_1/3']
# Train_File_Name='train_bullet_hole_v1'
# Test_File_Name='test_bullet_hole_v1'
#
# #1 train config net
# Class_Dic={'bullet_hole':1,'chest_bitmap':2}
# Config_path='Config_template'
# Config_template_file='ssdlite_mobilenet_v1_coco.config'
# image_w=640
# image_h=480
# train_num_step=200000
# train_batch_size=6
#
# #2 train data out config
# output_train_dir='bullet_hole_output_list'
#
# #3 model out config
# module_out_put_dir='model_'+output_train_dir
# model_ckpt=33426
#
# #4 run test config
# use_camer=False
# run_test_file_dir='run_test_data'
# run_test_file_out_dir=run_test_file_dir+'/'+'test_data_out/'
# imput_image_w=1280
# imput_image_h=960
#---------------------------------------------------




#1 train data prepare config
Train_Data_Path=['tool_train/1','tool_train/2','tool_train/3','tool_train/4','tool_train/5','tool_train/6']
# Train_Data_Path=['tool_train/7','tool_train/8']
Test_Data_Path=['tool_test/1','tool_test/2','tool_test/3','tool_test/4','tool_test/5','tool_test/6']
Train_File_Name='train_tool_v1'
Test_File_Name='test_tool_v1'



Class_Dic={'spanner':1,'adjustable_spanner':2,'fracture_spanner':3,
           'inner_hexagon_spanner':4,'peel tongs':5,'steel ruler':6,
           'bolt driver':7,'tongs':8}
Config_path='Config_template'
Config_template_file='ssdlite_mobilenet_v1_coco.config'
image_w=480
image_h=320
train_num_step=200000
train_batch_size=6

#2 train data out config
output_train_dir='tool_model_output_list'

#3 model out config
module_out_put_dir='model_'+output_train_dir
model_ckpt=31453

#4 run test config
use_camer=False
run_test_file_dir='run_test_data'
run_test_file_out_dir=run_test_file_dir+'/'+'test_data_out/'
imput_image_w=480
imput_image_h=320
#---------------------------------------------------
