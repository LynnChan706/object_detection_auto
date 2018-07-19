#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'ccchen_706@126.com'
'''

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from gconfig import *

train_path = Train_Data_Path
test_path = Test_Data_Path

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            if os.path.splitext(root.find('filename').text)[1] == '.jpg':
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
            else:
                value = (root.find('filename').text+'.jpg',
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def creat_csv():
    if type(train_path) !=list:
        xml_train = xml_to_csv(train_path)
        xml_train.to_csv(train_path+'/'+Train_File_Name+'.csv', index=None)
        print('Successfully converted train xml to csv.')
    else:
        for i in train_path:
            xml_train = xml_to_csv(i)
            xml_train.to_csv(i + '/' + Train_File_Name + '.csv', index=None)
        print('Successfully converted list train xml to csv.')

    if type(test_path) != list:
        xml_test = xml_to_csv(test_path)
        xml_test.to_csv(test_path+'/'+Test_File_Name+'.csv', index=None)
        print('Successfully converted test xml to csv.')
    else:
        for i in test_path:
            xml_train = xml_to_csv(i)
            xml_train.to_csv(i + '/' + Test_File_Name + '.csv', index=None)
        print('Successfully converted list train xml to csv.')

if __name__ == '__main__':
    creat_csv()

