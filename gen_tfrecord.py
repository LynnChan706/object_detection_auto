#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'ccchen_706@126.com'
'''



import os
import io
import pandas as pd
import tensorflow as tf
from gconfig import *

from PIL import Image
from collections import namedtuple, OrderedDict


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def class_text_to_int(row_label):
    if row_label in Class_Dic.keys():
        return Class_Dic[row_label]
    else:
        None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    # print(image)
    # image = image.resize((image_w, image_h))
    # if(image)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example

def creat_file(imgpath,csv_file_name,tf_file_name):
    if type(imgpath) != list:
        writer = tf.python_io.TFRecordWriter(tf_file_name)
        examples = pd.read_csv(imgpath+'/'+csv_file_name)
        grouped = split(examples, 'filename')
        for group in grouped:
            tf_example = create_tf_example(group, imgpath)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecords: {}'.format(tf_file_name))
    else:
        writer = tf.python_io.TFRecordWriter(tf_file_name)
        for i in imgpath:
            examples = pd.read_csv(i + '/' + csv_file_name)
            grouped = split(examples, 'filename')
            for group in grouped:
                # print(imgpath,i,group)
                tf_example = create_tf_example(group, i)
                writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created list the TFRecords: {}'.format(tf_file_name))


def crate_tfrecord():
    folder=os.path.exists(output_train_dir)
    if not folder:
        os.makedirs(output_train_dir)
    creat_file(Train_Data_Path,Train_File_Name+'.csv',output_train_dir+'/'+Train_File_Name+'.record')
    creat_file(Test_Data_Path, Test_File_Name+'.csv', output_train_dir+'/'+Test_File_Name+'.record')

def main(_):
    crate_tfrecord()

if __name__ == '__main__':
    tf.app.run(main=main)
