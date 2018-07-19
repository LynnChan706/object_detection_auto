#!/usr/bin/env python3.5
# coding=utf-8

'''
@date = '17/12/1'
@author = 'lynnchan'
@email = 'ccchen_706@126.com'
'''

from gconfig import *

def get_item_str(id,name):
    item_str="item {\n   id:"+str(id)+"\n   name: '"+name+"'\n"+"}\n"
    return item_str

def get_dic_str(data_dic):
    gstr=''
    for k in data_dic.keys():
        gstr =gstr + get_item_str(data_dic[k],k)
    return gstr

def write_pbtxt(pb_str,filename):
    fh = open(filename, 'w')
    fh.write(pb_str)
    fh.close()

def write_config_pbtxt():
    pb_str=get_dic_str(Class_Dic)
    write_pbtxt(pb_str,output_train_dir+'/'+output_train_dir+'.pbtxt')

if __name__ == '__main__':
    print(get_dic_str(Class_Dic))
    write_config_pbtxt()