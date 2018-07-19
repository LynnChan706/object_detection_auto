#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'ccchen_706@126.com'
'''

import os
from PIL import Image

from gconfig import *

train_path = Train_Data_Path
test_path = Test_Data_Path

def format_list(path):
    for i in os.listdir(path):
        dic_i = os.path.join(path, i)
        if os.path.isfile(dic_i) and (os.path.splitext(i)[1] == '.jpg' or os.path.splitext(i)[1] == '.JPG'):
            img = Image.open(dic_i).convert('RGB')
            img.save(dic_i)

def format_img():
    if type(train_path) !=list:
        format_list(train_path)
        print('Successfully format_img.')
    else:
        for i in train_path:
            format_list(i)
            print('Successfully format list img: '+i)

    if type(test_path) != list:
        format_list(test_path)
        print('Successfully format_img.')
    else:
        for i in test_path:
            format_list(i)
            print('Successfully format list img: '+i)

if __name__ == '__main__':
    format_img()

