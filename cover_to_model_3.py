#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'ccchen_706@126.com'
'''
import os
import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2
from gconfig import *
import shutil
slim = tf.contrib.slim

pipeline_config_path=os.path.abspath(output_train_dir).replace('\\', '/') + '/' + output_train_dir + '.config'
train_dir=os.path.abspath(output_train_dir)
trained_checkpoint_prefix=os.path.abspath(output_train_dir).replace('\\', '/') + '/model.ckpt-' +str(model_ckpt)

def main(_):

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge('', pipeline_config)
    input_shape = None
    exporter.export_inference_graph('image_tensor', pipeline_config,
                                    trained_checkpoint_prefix,
                                    module_out_put_dir, input_shape)

if __name__ == '__main__':
    folder=os.path.exists(module_out_put_dir)
    if not folder:
        os.makedirs(module_out_put_dir+'/')
    shutil.copy(train_dir+'/'+output_train_dir+'.pbtxt', module_out_put_dir)
    tf.app.run()
