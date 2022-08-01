    # -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 23:32:59 2022

@author: 磷Sandwich
"""
import pandas as pd
import random
from random import choice
data=[]


with open("raw_train-副本.txt", "r") as f:  # 打开文件
    for i in range(0,4008): #选择数据的行数
        data.append(f.readline())  # 读取文件
    
print(len(data))    

    
random.seed(10)

select=random.sample(data,800)#选择测试集数据数目
print(len(select))

#生成训练集
filename =open("raw_train.txt", 'w')
for i in data:
    filename.write(str(i))
#生成测试集
filename = open("raw_test.txt", "w")
for value in select:
    filename.write(str(value))

filename.close()



