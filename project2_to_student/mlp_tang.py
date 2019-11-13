import numpy as np
import torch as pt
import scipy.io as sio
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as fn
import torch.optim as opt
import torch.utils.data as data

pt.manual_seed(1)

# 加载训练集、验证集、测试集
data_np = sio.loadmat('./train_data.mat')['train_data'].astype(np.float32)
label_np = np.squeeze(sio.loadmat('./train_label.mat')['train_label'] == 1).astype(np.int64)
valid_index = range(0, 800, 8)
valid_data_np = data_np[valid_index, :]
valid_label_np = label_np[valid_index]
train_data_np = np.delete(data_np, valid_index, axis=0)
train_label_np = np.delete(label_np, valid_index, axis=0)
# 添加噪声，增加训练集样本数量
noise = np.random.randn(train_data_np.shape[0], 19 * 19).astype(np.float32) * 0.2
train_data_np_noise = noise + train_data_np
train_data_np = np.concatenate((train_data_np, train_data_np_noise), axis=0)
train_label_np = np.concatenate((train_label_np, train_label_np), axis=0)
test_data_np = sio.loadmat('./test_data.mat')['test_data'].astype(np.float32)

'''
# 归一化至标准正态分布
for i in range(train_data_np.shape[0]):
    train_data_np[i] = (train_data_np[i] - np.mean(train_data_np[i])) / np.std(train_data_np[i])
for i in range(valid_data_np.shape[0]):
    valid_data_np[i] = (valid_data_np[i] - np.mean(valid_data_np[i])) / np.std(valid_data_np[i])
for i in range(test_data_np.shape[0]):
    test_data_np[i] = (test_data_np[i] - np.mean(test_data_np[i])) / np.std(test_data_np[i])
'''

# np array to tensor
train_data = pt.from_numpy(train_data_np)
train_label = pt.from_numpy(train_label_np)
valid_data = pt.from_numpy(valid_data_np)
valid_label = pt.from_numpy(valid_label_np)
test_data = pt.from_numpy(test_data_np)


class MLP(nn.Module):
    def __init__(self, in_dim, n_hd1, n_hd2, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, n_hd1)
        self.fc2 = nn.Linear(n_hd1, n_hd2)
        self.fc3 = nn.Linear(n_hd2, out_dim)

    def forward(self, x):
        x = fn.relu(self.fc1(x))
        # x = fn.dropout(x, p=0.3)
        x = fn.relu(self.fc2(x))
        # x = fn.dropout(x, p=0.3)
        # x = pt.sigmoid(self.fc3(x))
        x = self.fc3(x)
        return x


mlp_net = MLP(19 * 19, 12, 6, 2)

data_set = data.TensorDataset(train_data, train_label)
loader = data.DataLoader(dataset=data_set, batch_size=10, shuffle=True)
optimizer = opt.ASGD(mlp_net.parameters(), lr=0.0016, weight_decay=0.027)
criterion = nn.CrossEntropyLoss()

loss_list = []
train_acc_list = []
valid_acc_list = []
iter_times = 80

# 训练
for epoch in range(iter_times):
    loss_list_tmp = []
    train_acc_list_tmp = []
    valid_acc_list_tmp = []
    for step, (batch_x, batch_y) in enumerate(loader):
        optimizer.zero_grad()
        output = mlp_net(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        print('loss: ', loss.item())
        loss_list_tmp.append(loss.item())
        viz_res = mlp_net(valid_data).detach().numpy()
        viz_output = np.argmax(viz_res, axis=1).astype(np.int)
        valid_acc_list_tmp.append((viz_output == valid_label_np).astype(np.int).mean())

        viz_res = mlp_net(train_data).detach().numpy()
        viz_output = np.argmax(viz_res, axis=1).astype(np.int)
        train_acc_list_tmp.append((viz_output == train_label_np).astype(np.int).mean())

    loss_list.append(np.mean(loss_list_tmp))
    valid_acc_list.append(np.mean(valid_acc_list_tmp))
    train_acc_list.append(np.mean(train_acc_list_tmp))


train_res = mlp_net(train_data).detach().numpy()
train_output = np.argmax(train_res, axis=1).astype(np.int)
print('Train Acc: ', (train_output == train_label_np).astype(np.int).mean())

valid_res = mlp_net(valid_data).detach().numpy()
valid_output = np.argmax(valid_res, axis=1).astype(np.int)
print('Valid Acc: ', (valid_output == valid_label_np).astype(np.int).mean())

mlp_net.zero_grad()
test_res = mlp_net(test_data).detach().numpy()
test_output = np.argmax(test_res, axis=1)

txt = open('test_label.txt', 'wt')
for index, res in enumerate(test_output):
    print(index + 1, res, file=txt)
txt.close()


# Loss曲线
plt.figure(1)
x1 = range(0, iter_times)
plt.plot(x1, loss_list, '.-')
plt.title('Train Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# Acc曲线
fig = plt.figure(2)
plt.title('Acc')
plt.ylim(0.4, 1)
x2 = range(0, iter_times)
plt.plot(x2, valid_acc_list, color='red', label='valid', ls='-.')
x3 = range(0, iter_times)
plt.plot(x3, train_acc_list, color='blue', label='train', ls=':')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()


