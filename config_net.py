#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'ccchen_706@126.com'
'''

import os
from gconfig import *
import google.protobuf
from object_detection.utils import config_util
from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.lib.io import file_io


def _save_pipeline_config(pipeline_config, directory,filename):
  if not file_io.file_exists(directory):
    file_io.recursive_create_dir(directory)
  pipeline_config_path = os.path.join(directory, filename)
  config_text = text_format.MessageToString(pipeline_config)
  with tf.gfile.Open(pipeline_config_path, "wb") as f:
    tf.logging.info("Writing pipeline config file to %s",
                    pipeline_config_path)
    f.write(config_text)


def creat_train_config_file():
    configs = config_util.get_configs_from_pipeline_file(Config_path+'/'+Config_template_file)

    configs['model'].ssd.num_classes=len(Class_Dic.keys())
    configs['model'].ssd.image_resizer.fixed_shape_resizer.height=image_h
    configs['model'].ssd.image_resizer.fixed_shape_resizer.width=image_w

    configs['train_config'].num_steps=train_num_step
    configs['train_config'].batch_size=train_batch_size
    configs['train_config'].fine_tune_checkpoint=''
    configs['train_config'].fine_tune_checkpoint_type=''
    configs['train_config'].from_detection_checkpoint=False

    configs['train_input_config'].tf_record_input_reader.input_path[0]=os.path.abspath(output_train_dir).replace('\\','/')+'/'+Train_File_Name+'.record'
    configs['train_input_config'].label_map_path=os.path.abspath(output_train_dir).replace('\\','/')+'/'+output_train_dir+'.pbtxt'

    configs['eval_input_config'].tf_record_input_reader.input_path[0]=os.path.abspath(output_train_dir).replace('\\','/')+'/'+Test_File_Name+'.record'
    configs['eval_input_config'].label_map_path=os.path.abspath(output_train_dir).replace('\\','/')+'/'+output_train_dir+'.pbtxt'

    pip_config=config_util.create_pipeline_proto_from_configs(configs)
    _save_pipeline_config(pip_config,output_train_dir,output_train_dir+'.config')

    print('net work config ok !')


if __name__ == '__main__':
    creat_train_config_file()