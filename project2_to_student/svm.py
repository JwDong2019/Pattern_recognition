# -*- coding: UTF-8 -*-
import sys
path=r'C:\Users\zhtjw\Desktop\libsvm-3.24\python'
sys.path.append(path)
from svmutil import *
from scipy.io import loadmat
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

train_label = loadmat(r'train_label.mat')
train_label = list(train_label['train_label'])
train_data = loadmat(r'train_data.mat')
train_data = list(train_data['train_data'])
test_data = loadmat(r'test_data.mat')
test_data = list(test_data['test_data'])
test_label = loadmat(r'test_label.mat')
test_label = list(test_label['test_label'])
acc = np.zeros(shape=(6,6))
for i in range(6):
    for j in range(6):
        g_d = math.pow(2,(2*i-5))
     #   g_d = i/10+0.01
     #   c_d = j/10+0.01
        c_d = math.pow(2,(2*j-5))
        options = '-t 3 -d 3 -c %f -g %f'%(c_d ,g_d)
        model = svm_train(train_label,train_data,options)
        print("result:")
        p_label, p_acc, p_val = svm_predict(test_label, test_data, model)
        corr = 0
        print(len(p_label))
        for k in range(len(p_label)):
      #print(p_label[k])
           #   f = open(r'SVM.txt','a')
            #  f.write(str(k+1)+' '+str(int(p_label[k]))+'\n')
           #   f.close()
              if p_label[k] == test_label[k]:
                 corr = corr+1
              acc[i,j] = corr/len(p_label)

f, ax = plt.subplots(figsize=(6, 6))
ax = sns.heatmap(acc, annot=True,cmap="YlGnBu",linewidths=.5,annot_kws={'size':10,'weight':'bold','color':'red'})
ax.set_title('Sigmod')
ax.set_xlabel('lb g')
ax.set_xticklabels([]) #设置x轴图例为空值
ax.set_ylabel('lb b')
ax.set_yticklabels([])
plt.show()
