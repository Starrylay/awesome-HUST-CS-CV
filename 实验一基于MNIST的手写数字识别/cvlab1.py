import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import numpy as np
learning_rate = 1e-4
keep_prob_rate = 0.2 #
max_epoch = 20
BATCH_SIZE = 50

DOWNLOAD_MNIST = False

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("using cuda")
else:
    device = torch.device('cpu')
    print("using cuda")

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True
train_data = torchvision.datasets.MNIST(root='./mnist/',train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST,)
train_loader = Data.DataLoader(dataset = train_data ,batch_size= BATCH_SIZE ,shuffle= True)

test_data = torchvision.datasets.MNIST(root = './mnist/',train = False)
test_x = Variable(torch.unsqueeze(test_data.data,dim  = 1)).type(torch.cuda.FloatTensor)[:500]/255.
test_y = test_data.targets[:500].numpy()

test_y_list=np.zeros(10)
for item in test_y:
    test_y_list[item]+=1

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            # nn.Sigmoid(),
            #nn.PReLU(),
            nn.ReLU(),  # activation function
            nn.MaxPool2d(2),  # pooling operation
        )
        self.conv2 = nn.Sequential(

            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        ##3 4 replace 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            # nn.Sigmoid(),
            #nn.PReLU(),
            nn.ReLU(),
            nn.MaxPool2d(2), # pooling operation
        )
        self.conv4 = nn.Sequential(

            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
        self.conv5 = nn.Sequential(

            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=1,
                stride=2,
                padding=0,
            ),
        )

        self.out1 =nn.Sequential(
            nn.Linear(12544, 1024, bias=True), # full connect
             nn.ReLU()
            # nn.Sigmoid()
            #nn.PReLU()
        )

        self.dropout = nn.Dropout(keep_prob_rate)
        self.out2 = nn.Linear(1024, 10, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)#包含 relu 和 最大池化

        x2_1 = self.conv2(x1)
        x2_1 = F.relu(x2_1)

        x2_1 = self.conv2(x2_1)
        x2_1 = F.relu(x1 + x2_1)

        x2_2 = self.conv2(x2_1)
        x2_2 = F.relu(x2_2)
        x2_2 = self.conv2(x2_2)
        x2_2 = F.relu(x2_1 + x2_2)

        x3_1 = self.conv3(x2_2)# 包含 relu 和 最大池化
        x3_1 = self.conv4(x3_1)
        x2_2 = self.conv5(x2_2)# 通道数转换64-256
        x3_1 = F.relu(x3_1 + x2_2)

        x3_2 = self.conv4(x3_1)
        x3_2 = F.relu(x3_2)
        x3_2 = self.conv4(x3_2)
        x3_2 = F.relu(x3_1 + x3_2)

        x3_2 = x3_2.view(x3_2.size(0), 12544)  # flatten the output 数字可以计算，也可以运行报错根据错误计算（简单）
        out1 = self.out1(x3_2) # full connect1

        out1 = self.dropout(out1)
        out2 = self.out2(out1)
        output = F.softmax(out2)
        return output


def test(cnn):#返回F1-score 和 accuracy
    predict = np.ones(10)# to avoid 0 being denominator(分母)
    TP = np.zeros(10)
    y_pre = cnn(test_x)
    _, pre_index = torch.max(y_pre, 1)#max return first is the max value of the line,the second is the index of the max value in the line.
    pre_index = pre_index.view(-1)
    pre_index_cpu = pre_index.cpu()
    prediction = pre_index_cpu.data.numpy()
    for index, item in enumerate(prediction):
        predict[item]+=1
        if prediction[index]==test_y[index]:
            TP[item]+=1
    precision = TP/predict
    recall = TP/test_y_list
    F1 = 2*precision*recall/(precision+recall)
    correct = np.sum(prediction == test_y)
    correct = correct/500
    return F1, correct


def train(cnn):
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(max_epoch):
        for step, (x_, y_) in enumerate(train_loader):
            x, y = Variable(x_).to(device), Variable(y_).to(device)
            output = cnn(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()# clear the parameters such as weights and bias, for the function 'backward' would add the new to the last ones.
            loss.backward()
            optimizer.step()# conduct the gradient descent using the gradient generated in the 'backward'.

        F1, correct = test(cnn)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print("=" * 10, "epoch:",epoch, "=" * 5, "test F1 is ", F1, "=" * 5, "test sum accuracy is ", correct)
if __name__ == '__main__':

    cnn = CNN()
    cnn.to(device)
    train(cnn)
