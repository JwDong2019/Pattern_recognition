import torch
import torch.nn.functional as F   
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from scipy.io import loadmat

n_epochs = 500     
batch_size = 20  

train_data = loadmat(r'train_data.mat')['train_data']
train_label = loadmat(r'train_label.mat')['train_label']
test_data = loadmat(r'test_data.mat')['test_data']
test_label = loadmat(r'test_label.mat')['test_label']
train_label = (train_label+1)/2

class MLP(torch.nn.Module):   
    def __init__(self):
        super(MLP,self).__init__()    
        self.fc1 = torch.nn.Linear(361,512)  
        #self.norm1 = torch.nn.BatchNorm1d(512,momentum=0.5)
        self.fc2 = torch.nn.Linear(512,128)  
        #self.norm2 = torch.nn.BatchNorm1d(128,momentum=0.5)
        self.fc3 = torch.nn.Linear(128,2)
        self.drop = torch.nn.Dropout(0.5)

        
    def forward(self,din):
        din = din.unsqueeze(0)
        din = din.view(len(din),-1)       
        dout = F.relu(self.fc1(din))   
        dout = F.relu(self.fc2(dout))
        dout = F.softmax(self.fc3(dout),dim=1)  
        return dout

def train():
    lossfunc = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01)

    for epoch in range(n_epochs):
        train_loss = 0.0
        for i in range(len(train_label)):
            optimizer.zero_grad()  
            output = model(torch.Tensor(train_data[i]))    
            tar = np.zeros(2)
            #print(tar)
            if train_label[i]==1:
                tar[0] = 1
            else:
                tar[1] = 1
            loss = criterion(output.squeeze(0),torch.Tensor(tar))
            
            loss.backward()         
            optimizer.step()        
            train_loss += loss.item()*torch.Tensor(train_data).size(0)
        train_loss = train_loss / len(train_label)
        print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss))
    test()

def test():
    correct = 0
    total = 0
    lab = np.zeros(len(test_data))
    with torch.no_grad():  
        for j in range(len(test_data)):
            
            outputs = model(torch.Tensor(test_data[j]))
            outputs = outputs.squeeze(0)
            if outputs[0]>outputs[1]:
                lab[j]=1
            f = open(r'mlp.txt','a')
            f.write(str(j+1)+' '+str(int(lab[j])*2-1)+'\n')
            f.close()
            if int(lab[j]) == (int(test_label[j].squeeze(0))+1)/2:
                correct = correct+1
        acc = correct/len(test_data)
    print(lab)
    print(acc)

model = MLP()
if __name__ == '__main__':
    train()
