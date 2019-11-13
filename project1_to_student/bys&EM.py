from scipy.io import loadmat
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import cv2
import math
############Load and PreProcessing Image########

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

msk = loadmat('Mask.mat')
msk = list(msk['Mask'])
msk = np.array(msk)
msk_r = np.random.rand(240,320,3)
for i in range(2):
    msk_r[:,:,i] = msk
img = mpimg.imread('309.bmp')

img_g = np.array(rgb2gray(img))
bef_g = img_g*msk
#bef_gs = Image.fromarray(bef_g)
#plt.imshow(bef_gs)
#plt.show()
#bef_r = np.random.rand(240,320,3)
img_r = np.array(img)
bef_r = img_r*msk_r
#bef_rs = Image.fromarray(np.uint8(bef_r))
#plt.imshow(bef_rs)
#plt.show()
############Gray bys##########

datasets = loadmat('array_sample.mat')
datasets = datasets['array_sample']
train_data = np.array(datasets[:,0],dtype='float32')
train_label = np.array(datasets[:,4],dtype='int32')
bayer = cv2.ml.NormalBayesClassifier_create()
ret = bayer.train(train_data,cv2.ml.ROW_SAMPLE,train_label)

height = bef_g.shape[0]
weight = bef_g.shape[1]
aft_g = np.zeros(shape=(240,320))
for i in range(height):
    for j in range(weight):
        pt = np.array(bef_g[i,j]/256,dtype='float32')
        (ret, res) = bayer.predict(pt)
        aft_g[i,j] = (res+1)/2

aft_g = Image.fromarray(aft_g*256*msk)
plt.subplot(221)
plt.title('gray bys')
plt.imshow(aft_g)
#plt.show()

###########RGB bys##########
train_data = np.array(datasets[:,1:4],dtype='float32')
train_label = np.array(datasets[:,4],dtype='int32')
bayer = cv2.ml.NormalBayesClassifier_create()
ret = bayer.train(train_data,cv2.ml.ROW_SAMPLE,train_label)
aft_r = np.zeros(shape=(240,320))
for i in range(240):
        pt = np.array(bef_r[i,:,:],dtype='float32')
        (ret, res) = bayer.predict(pt)
        for j in range(320):
            aft_r[i,j] = (res[j]+1)/2
aft_r = Image.fromarray(aft_r*256*msk)
plt.subplot(222)
plt.title('RGB bys')
plt.imshow(aft_r)
#plt.show()

############Gray EM##########
X = np.zeros(240*320)
for i in range(240):
        for j in range(320):
            X[i*320+j] = bef_g[i,j]

def ini_data(Sigma,k,N):
    global X
    global Mu
    global Expectations
    Mu = np.random.random(2)
    Expectations = np.zeros((N,k))
    print(X)
    print(max(X[:]))
    print(X.shape)

def e_step(Sigma,k,N):
    global Expectations
    global Mu
    global X
    for i in range(0,N):
        Denom = 0
        if X[i]>1:
            for j in range(0,k):
                Denom += math.exp((-1/(2*(float(Sigma**2))))*(float(X[i]-Mu[j]))**2)
            for j in range(0,k):
                Numer = math.exp((-1/(2*(float(Sigma**2))))*(float(X[i]-Mu[j]))**2)
                Expectations[i,j]=Numer/(Denom+0.000001)

def m_step(k,N):
    global Expectations
    global X
    for j in range(0,k):
        Numer = 0
        Denom = 0
        for i in range(0,N):
            if X[i]>1:
                
                Numer += Expectations[i,j]*X[i]
                Denom +=Expectations[i,j]
        Mu[j] = Numer / Denom
        print(Mu)

Sigma =10
k=2
N=240*320
iter_num=10
ini_data(Sigma,k,N)
for i in range(iter_num):
        e_step(Sigma,k,N)
        m_step(k,N)
after_g = np.zeros(shape=(240,320))
order=np.zeros(240*320)
#plt.hist(order,bins=100)
#plt.show()
for i in range(240*320):
    if X[i]>1:
     for j in range(2):
         if Expectations[i,j]==max(Expectations[i,:]):
            order[i]=j+1     
for i in range(240):
    for j in range(320):
        after_g[i,j] = order[i*320+j]


after_g = Image.fromarray(after_g*128)
plt.subplot(223)
plt.title('gray EM')
plt.imshow(after_g)
#plt.show()

############RGB EM##########
X = np.zeros((240*320,3))
for i in range(240):
        for j in range(320):
            for k in range(3):
                X[i*320+j,k] = bef_r[i,j,k]
def ini_data(Sigma,k,N):
    global X
    global Mu
    global Expectations
    global Expectation
    Mu = np.random.random((2,3))
    Expectations = np.zeros(((N,k,3)))
    Expectation = np.zeros((N,k))

def e_step(Sigma,k,N):
    global Expectations
    global Mu
    global X
    global Expectation
    for i in range(0,N):
        Denom = np.zeros(3)
        Denom_ = 0
        Numer_ = 0
        Numer= np.zeros(3)
        for j in range(0,k):
            Denom_ += math.exp((-1/(2*(float(Sigma**2))))*np.dot((X[i,:]-Mu[j,:]),(X[i,:]-Mu[j,:])))
        for j in range(0,k):
            Numer_ = math.exp((-1/(2*(float(Sigma**2))))*np.dot((X[i,:]-Mu[j,:]),(X[i,:]-Mu[j,:])))
            Expectation[i,j]=Numer_/(Denom_+0.000001)
        for s in range(3):
            if X[i,0]>1:
                for j in range(0,k):
                    Denom[s] += math.exp((-1/(2*(float(Sigma**2))))*np.dot((X[i,s]-Mu[j,s]),(X[i,s]-Mu[j,s])))
                for j in range(0,k):
                    Numer[s] = math.exp((-1/(2*(float(Sigma**2))))*np.dot((X[i,s]-Mu[j,s]),(X[i,s]-Mu[j,s])))
                    Expectations[i,j,s]=Numer[s]/Denom[s]

def m_step(k,N):
    global Expectations
    global X
    global Expectation
    for j in range(0,k):
        Denom = np.zeros(3)
        Numer= np.zeros(3)
        for i in range(0,N):
            
            for s in range(3):
                if X[i,0]>1:
                    Numer[s] += Expectations[i,j,s]*X[i,s]
                    Denom[s] +=Expectations[i,j,s]
        Mu[j,s] = Numer[s] / Denom[s]
        print(Mu)
        
def run(Sigma,k,N,iter_num):
    ini_data(Sigma,k,N)
    for i in range(iter_num):
        e_step(Sigma,k,N)
        m_step(k,N)


run(10,2,240*320,10)
after_g = np.zeros(shape=(240,320))
order=np.zeros(240*320)
for i in range(240*320):
    for j in range(2):
        for s in range(3):
            if X[i,0]>1:
               if Expectation[i,j]==max(Expectation[i,:]):
                         order[i]=j+1     
for i in range(240):
    for j in range(320):
        for s in range(3):
            after_g[i,j] = order[i*320+j]
after_ge = Image.fromarray(after_g*128)
plt.subplot(224)
plt.title('RGB EM')
plt.imshow(after_ge)
plt.show()
