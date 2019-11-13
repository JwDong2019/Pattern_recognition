from libsvm.python.svmutil import *
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n = 120

# # # linear
# x1 = np.random.normal(0, 1, (n, 2))
# y1 = np.ones(n, dtype=np.int)
#
# x2 = np.random.normal(5, 1, (n, 2))
# y2 = -np.ones(n, dtype=np.int)

# # nonlinear
x1 = np.random.normal(0, 1, (n, 2))
y1 = np.ones(n, dtype=np.int)

x2 = np.random.normal(3, 1, (n, 2))
y2 = -np.ones(n, dtype=np.int)


x = np.concatenate((x1, x2), 0)
y = np.concatenate((y1, y2), 0)
param = '-s 0 -t 0 -c 1 -q'
model = svm_train(y, x, param)

# visualize
plt.scatter(x1[:, 0], x1[:, 1], s=40, c='g',marker='D')
plt.scatter(x2[:, 0], x2[:, 1], s=40, c='m',marker='D')

support_vector_coefficients = model.get_sv_coef()
sv_idx = model.get_sv_indices()

weight = np.zeros([1, 2], np.float32)
# for i, idx in enumerate(sv_idx):
#     plt.scatter(x[idx - 1, 0], x[idx - 1, 1], s=40, c='g', marker='D')

for i, idx in enumerate(sv_idx):
    if y[idx - 1] == 1:
        plt.scatter(x[idx - 1, 0], x[idx - 1, 1], s=40, c='g',marker='D')
    else:
        plt.scatter(x[idx - 1, 0], x[idx - 1, 1], s=40, c='m',marker='D')
    weight += support_vector_coefficients[i] * x[idx - 1]
bias = -model.rho.contents.value

a = -weight[0, 0] / weight[0, 1]
xx = np.linspace(-3, 13)
yy = a * xx - bias / weight[0, 1]
margin = 1 / np.sqrt(np.sum(weight ** 2))
yy_down = yy - np.sqrt(1 + a ** 2) * margin
yy_up = yy + np.sqrt(1 + a ** 2) * margin

plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.title('nonlinear')
plt.tick_params(labelsize=15)
plt.xlim([-2, 6])#nonlinear plt.xlim([-2, 6]).linear plt.xlim([-2, 10])
plt.ylim([-2, 6])
plt.show()
