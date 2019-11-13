# coding:UTF-8

import random
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

# calculate mean and covariation
def parameters(numbers, labels, tar):
    numbers = numbers.transpose()

    mean = np.mean(numbers[:, labels == tar],axis = 1)
    covariation = np.cov(numbers[:, labels == tar])

    return mean, covariation

# Assuming that the attributes are gaussian, compute the probability density function
def calculateProbability(data, x, mean, covariation):

    x = x.reshape([-1, data.shape[1]])
    if len(mean) >= 2:
        a = np.mat(x - mean)
        b = np.mat(np.linalg.inv(covariation))
        c = a * b
        d = np.sum(np.multiply(c , a), axis = -1)
        exponent = np.exp((-1/2) * d)
        pro = (1 / (np.sqrt(((2 * np.pi) ** len(mean)) * np.linalg.det(covariation)))) * exponent

    elif len(mean) == 1:
        exponent = np.exp((-np.multiply(((x - mean)*(1/covariation)),(x - mean)) / 2 ))
        pro = (1 / (np.sqrt(((2 * np.pi) ** len(mean)) * covariation))) * exponent

    return pro



def calculateClassProbabilities(data, x, mean1, covariation1, mean2, covariation2, labels):
    # mean1, covariation1 = parameters(data, labels, 1)
    # mean2, covariation2 = parameters(data, labels, -1)
    shape = x.shape
    pro_pos =(list((labels == 1).astype(int)).count(1)) / len(labels)
    pro_neg = 1 - pro_pos
    p1 = calculateProbability(data, x, mean1, covariation1)*pro_pos
    p2 = calculateProbability(data, x, mean2, covariation2)*pro_neg

    pre = p1 > p2
    pre = pre.astype(np.uint8) * 128 + 127
    pre = np.asarray(pre).reshape(shape[0:2])
    return pre

    ########  E Step   ######
def E_step(data, x, mean1, covariation1, mean2, covariation2, alpha1, alpha2):

    pro1 = alpha1 * calculateProbability(data, x, mean1, covariation1)
    pro2 = alpha2 * calculateProbability(data, x, mean2, covariation2)
    sum_pro = pro1 + pro2 + 1e-3

    gamma1 = pro1 / sum_pro
    gamma2 = pro2 / sum_pro

    return gamma1, gamma2

    ########   M Step  ######
def M_step(data, x, mean1, mean2, gamma1, gamma2):

    gamma1 = np.array(gamma1)
    gamma2 = np.array(gamma2)
    gamma1 = gamma1.squeeze()
    gamma2 = gamma2.squeeze()
    x = x.reshape([-1, data.shape[1]])
    mean1_new = np.dot(gamma1, x) / np.sum(gamma1)
    mean2_new = np.dot(gamma2, x) / np.sum(gamma2)

    # covariation1_new = np.sqrt(np.dot(gamma1*((x - mean1).transpose()),(x - mean1)) / np.sum(gamma1))
    # covariation2_new = np.sqrt(np.dot(gamma2*((x - mean2).transpose()),(x - mean2)) / np.sum(gamma2))
    covariation1_new = np.dot(gamma1*((x - mean1).transpose()),(x - mean1)) / np.sum(gamma1)
    covariation2_new = np.dot(gamma2*((x - mean2).transpose()),(x - mean2)) / np.sum(gamma2)


    alpha1_new = np.sum(gamma1) / len(gamma1)
    alpha2_new = np.sum(gamma2) / len(gamma2)

    return mean1_new, mean2_new, covariation1_new, covariation2_new, alpha1_new, alpha2_new

def em_train(data, x, labels, iter):

    alpha1 = np.random.rand()
    alpha2 = np.random.rand()
    # mean1, covariation1 = parameters(data, labels, 1)
    # mean2, covariation2 = parameters(data, labels, -1)

    if data.shape[1] == 3:

        mean1 = (np.array(np.random.rand(data.shape[1],1))).squeeze()
        covariation1 = np.array(np.random.rand(data.shape[1],data.shape[1]))
        mean2 = (np.array(np.random.rand(data.shape[1],1))).squeeze()
        covariation2 = np.array(np.random.rand(data.shape[1],data.shape[1]))

    elif data.shape[1] == 1:
        mean1 = np.array(np.random.rand(1))
        covariation1 = np.array(np.random.rand())
        mean2 = np.array(np.random.rand(1))
        covariation2 = np.array(np.random.rand())

    # gamma1, gamma2 = E_step(data, x, mean1, covariation1, mean2, covariation2, alpha1, alpha2)
    # mean1, mean2, covariation1, covariation2, alpha1, alpha2 = M_step(data, x, mean1, mean2, gamma1, gamma2)

    step = 0
    while (step < iter):
        step += 1
        gamma1, gamma2 = E_step(data, x, mean1, covariation1, mean2, covariation2, alpha1, alpha2)
        mean1, mean2, covariation1, covariation2, alpha1, alpha2 = M_step(data, x, mean1, mean2, gamma1, gamma2)

    return mean1, mean2, covariation1, covariation2, alpha1, alpha2


def main():
    filename = '/home/djw/PycharmProjects/project1_to student/array_sample.mat'
    maskname = '/home/djw/PycharmProjects/project1_to student/Mask.mat'
    testname = '/home/djw/PycharmProjects/project1_to student/309.bmp'
    dataset = scio.loadmat(filename)['array_sample']
    mask = scio.loadmat(maskname)['Mask']
    img = plt.imread(testname) / 255
    gra_data = np.expand_dims(dataset[:, 0], -1)
    rgb_data = dataset[:,1:4]
    labels = dataset[:,4].astype(np.int64)
    grayscale_image = 0.299 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
    #grayscale_image = cv2.imread('309.bmp',0)/255

    plt.figure(1)
    plt.subplot(1,3,1)
    plt.title('origin')
    plt.imshow(img)
    mean1, covariation1 = parameters(gra_data, labels, 1)
    mean2, covariation2 = parameters(gra_data, labels, -1)
    prediction_gray = calculateClassProbabilities(gra_data, grayscale_image, mean1, covariation1, mean2, covariation2, labels)
    prediction_gray = np.multiply(prediction_gray, mask)
    plt.subplot(1,3,2)
    plt.title('grey')
    plt.imshow(prediction_gray)

    mean1, covariation1 = parameters(rgb_data, labels, 1)
    mean2, covariation2 = parameters(rgb_data, labels, -1)

    prediction_rgb = calculateClassProbabilities(rgb_data, img, mean1, covariation1, mean2, covariation2, labels)
    prediction_rgb = np.multiply(prediction_rgb, mask)
    plt.subplot(1,3,3)
    plt.title('rgb')
    plt.imshow(prediction_rgb)

    step = 1
    plt.figure(2)
    while (step < 10):
        mean1, mean2, covariation1, covariation2, alpha1, alpha2 = em_train(gra_data, grayscale_image, labels, 200)
        prediction_em = calculateClassProbabilities(gra_data, grayscale_image, mean1, covariation1, mean2, covariation2, labels)
        prediction_em = np.multiply(prediction_em, mask)
        plt.subplot(3, 3, step)
        step = step + 1
        plt.title('em_gray')
        plt.imshow(prediction_em)


    plt.figure(3)
    step = 1
    while (step < 10):
        mean1, mean2, covariation1, covariation2, alpha1, alpha2 = em_train(rgb_data, img, labels, 200)
        prediction_em = calculateClassProbabilities(rgb_data, img, mean1, covariation1, mean2, covariation2, labels)
        prediction_em = np.multiply(prediction_em, mask)
        plt.subplot(3, 3, step)
        step = step + 1
        plt.title('em_rgb')
        plt.imshow(prediction_em)

    plt.show()

if __name__ == '__main__':
    main()
