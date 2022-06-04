import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import random
import argparse

learning_rate = 1e-4
keep_prob_rate = 0.2 #
max_epoch = 10
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

# print(type(train_data[0][0]))
# print(type(train_data[0][1]))
# print(train_data[0])
train_loader = Data.DataLoader(dataset = train_data ,batch_size= BATCH_SIZE ,shuffle= True)

test_data = torchvision.datasets.MNIST(root = './mnist/',train = False)
test_x = Variable(torch.unsqueeze(test_data.data,dim  = 1)).type(torch.cuda.FloatTensor)[:500]/255.
test_y = test_data.targets[:500].numpy()
test_y_tensor = torch.from_numpy(test_y)
test_y_tensor = Variable(test_y_tensor).to(device)
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
        output = F.softmax(out2, dim = 1)
        return  out2, output


def test(cnn):#返回F1-score 和 accuracy
    predict = np.ones(10)# to avoid 0 being denominator(分母)
    TP = np.zeros(10)

    _, y_pre = cnn(test_x)
    _, pre_index = torch.max(y_pre, 1)#max return first is the max value of the line,the second is the index of the max value in the line.
    # pre_index = pre_index.view(-1)
    pre_index_cpu = pre_index.cpu()
    prediction = pre_index_cpu.data.numpy()
    for index, item in enumerate(prediction):
        predict[item]+=1
        if prediction[index]==test_y[index]:
            TP[item]+=1
    precision = TP/predict
    recall = TP/test_y_list
    F1 = 2*precision*recall/(precision+recall+1e-3)
    correct = np.sum(prediction == test_y)
    correct = correct/len(test_y)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print("=" * 5, "test F1 is ", F1, "=" * 5, "test sum accuracy is ", correct)
    return F1, correct



def generate_adversarial_pattern(input_image, image_label, model, loss_func):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logit, prediction = model(input_image)
    loss = loss_func(prediction, image_label)
    #每次backward前清除上一次loss相对于输入的梯度
    if input_image.grad != None:
        input_image.grad.data.zero_()
    loss.backward()
    gradient = input_image.grad
    #每次backward后清除参数梯度，防止产生其他影响
    optimizer.zero_grad()
    #得到梯度的方向
    signed_grad = torch.sign(gradient)
    return signed_grad

def attack_fgsm(input_image, image_lable, model, loss_func , eps=0.01):
    #预测原来的样本类别
    # input_image = np.array([input_image])
    # input_image = torch.from_numpy(input_image)
    _, y_pre = model(input_image)
    pre_prob, pre_index = torch.max(y_pre, 1) #概率 和 类别
    #生成对抗样本
    # loss_func = nn.CrossEntropyLoss()
    input_image.requires_grad = True
    adv_pattern = generate_adversarial_pattern(input_image, image_lable,
                                                       model, loss_func)
    clip_adv_pattern = torch.clamp(adv_pattern, 0., 1.)
    perturbed_img = input_image + (eps * adv_pattern)
    perturbed_img = torch.clamp(perturbed_img, 0., 1.)
    #预测对抗样本的类别
    _, y_adv_pre = model(perturbed_img)
    adv_pre_prob, adv_pre_index = torch.max(y_adv_pre, 1)  # 概率 和 类别

    #可视化
    if args.is_view == True:

        fig, ax = plt.subplots(1,3,figsize=(20, 4))

        ax[0].imshow(input_image[0][0].cpu().detach().numpy().squeeze(), cmap = 'gray')
        ax[0].set_title('orignal sample\nTrue:{}  Pred:{}  Prob:{:.3f}'.format(image_lable[0].cpu().detach().numpy(), pre_index[0].cpu().detach().numpy(),  pre_prob[0].cpu().detach().numpy()))

        ax[1].imshow(clip_adv_pattern[0][0].cpu().detach().numpy().squeeze(), cmap='gray')
        ax[1].set_title(r'Adversarial Pattern - EPS: {}/255'.format(args.epsfenzi))

        ax[2].imshow(perturbed_img[0][0].cpu().detach().numpy().squeeze(), cmap='gray')
        ax[2].set_title('Attack sample\nTrue:{}  Pred:{}  Prob:{:.3f}'.format(image_lable.cpu().detach().numpy(), adv_pre_index[0].cpu().detach().numpy(), adv_pre_prob[0].cpu().detach().numpy()))

    if pre_index == image_lable and adv_pre_index != image_lable:

        if args.is_view == True:
            plt.savefig(r'D:\CV\cvlab\image\randomed\{}to{}eps{}.png'.format(image_label[0].cpu().detach().numpy(),
                                                                             adv_pre_index[0].cpu().detach().numpy(),
                                                                             args.epsfenzi), bbox_inches='tight')
            plt.show()
        return 1
    else:
        if args.is_view == True:
            plt.show()
        return 0

def targeted_fgsm(input_image, image_label, target_label, model, eps=0.01):
    # 预测原来的样本类别
    _, y_pre = model(input_image)
    pre_prob, pre_index = torch.max(y_pre, 1)  # 概率 和 类别
    loss_func = nn.CrossEntropyLoss()
    # 生成对抗样本
    input_image.requires_grad = True
    adv_pattern = generate_adversarial_pattern(input_image, target_label, model, loss_func)
    clip_adv_pattern = torch.clamp(adv_pattern, 0., 1.)
    perturbed_img = input_image - (eps * adv_pattern)
    perturbed_img = torch.clamp(perturbed_img, 0., 1.)
   # perturbed_img = F.softmax(perturbed_img, dim = 1)
    #perturbed_img = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    # 预测对抗样本的类别
    _, y_adv_pre = model(perturbed_img)
    adv_pre_prob, adv_pre_index = torch.max(y_adv_pre, 1)  # 概率 和 类别
    # 可视化
    if  args.is_view == True:
        fig, ax = plt.subplots(1, 3, figsize=(20, 4))

        ax[0].imshow(input_image[0][0].cpu().detach().numpy().squeeze(), cmap='gray')
        ax[0].set_title('orignal sample\nTrue:{}  Pred:{}  Prob:{:.3f}'.format(image_label[0].cpu().detach().numpy(),
                                                                               pre_index[0].cpu().detach().numpy(),
                                                                               pre_prob[0].cpu().detach().numpy()))

        ax[1].imshow(clip_adv_pattern[0][0].cpu().detach().numpy().squeeze(), cmap='gray')
        ax[1].set_title(r'Adversarial Pattern - EPS: {}/255'.format(args.epsfenzi))

        ax[2].imshow(perturbed_img[0][0].cpu().detach().numpy().squeeze(), cmap='gray')
        ax[2].set_title('Attack sample\nTrue:{}  Pred:{}  Prob:{:.3f}'.format(image_label[0].cpu().detach().numpy(),
                                                                              adv_pre_index[0].cpu().detach().numpy(),
                                                                              adv_pre_prob[0].cpu().detach().numpy()))

    if pre_index == image_label and adv_pre_index == target_label:
        if args.is_view == True:
            plt.savefig(r'D:\CV\cvlab\image\targeted\{}to{}eps{}.png'.format(image_label[0].cpu().detach().numpy(), adv_pre_index[0].cpu().detach().numpy(), eps), bbox_inches='tight')
            plt.show()
        return 1
    else:
        if args.is_view == True:
            plt.show()
        return 0

def targeted_loss(logit, target_label):
    y_based = torch.ones(10).to(device)
    y_based[target_label] = -10
    logit = logit.squeeze()
    loss = logit*y_based
    loss = torch.sum(logit*y_based)
    return loss

def generate_perturbed_images(input_images, image_labels, model, loss_func ,eps ):
    _, y_pres = model(input_images)
    input_images.requires_grad = True
    #生成梯度方向
    adv_patterns = generate_adversarial_pattern(input_images, image_labels, model, loss_func)
    perturbed_imgs = input_images + (eps * adv_patterns)
    perturbed_imgs = torch.clamp(perturbed_imgs, 0., 1.)
    return perturbed_imgs

def train(cnn):
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(max_epoch):
        for step, (x_, y_) in enumerate(train_loader):
            x, y = Variable(x_).to(device), Variable(y_).to(device)
            _,output = cnn(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()# clear the parameters such as weights and bias, for the function 'backward' would add the new to the last ones.
            loss.backward()
            optimizer.step()# conduct the gradient descent using the gradient generated in the 'backward'.
        F1, correct = test(cnn)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print("=" * 10, "epoch:",epoch, "=" * 5, "test F1 is ", F1, "=" * 5, "test sum accuracy is ", correct)

    f = open(r'./saved_model/cnn.pickle', 'wb+')
    pickle.dump(cnn, f)
    f.close()
    print("cnn is saved in ./saved_model/cnn.pickle")

def retrain(cnn):
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    for step, (x_, y_) in enumerate(train_loader):
        x, y = Variable(x_).to(device), Variable(y_).to(device)
        perturbed_x = generate_perturbed_images(x, y, cnn, loss_func, args.eps )
        output = cnn(perturbed_x)
        loss = loss_func(output, y)
        optimizer.zero_grad()  # clear the parameters such as weights and bias,
                               # for the function 'backward' would add the new to the last ones.
        loss.backward()
        optimizer.step()  # conduct the gradient descent using the gradient generated in the 'backward'.
    #将对抗训练后的样本存入本地文件
    f = open(r'./saved_model/retrain_cnn.pickle', 'wb+')
    pickle.dump(cnn, f)
    f.close()
    return cnn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_view", type=bool, default=True)
    parser.add_argument("--is_train", type=bool, default=False)#是否训练模型（否则从本地下载）
    parser.add_argument("--is_test", type=bool, default=True)
    parser.add_argument("--is_attack", type=bool, default=False)#是否攻击
    parser.add_argument("--is_adv_train", type=bool, default=False)#是否对抗训练
    parser.add_argument("--is_retrain", type=bool, default=False)#是否重新训练对抗模型（否则从本地下载）
    parser.add_argument("--eps", type=float, default=0.01)
    parser.add_argument("--epsfenzi", type=float, default= 1.0)
    args = parser.parse_args()
    cnn = CNN()
    cnn.to(device)
#######################################
    args.is_view = False
    args.is_train = False  #是否训练模型（否则从本地下载）
    args.is_test = False    #是否测试
    args.is_attack = True  #是否攻击
    fgsm_type = "randomed" # 攻击类型  "randomed"  or  "targeted"
    args.epsfenzi = 50.0

    args.is_adv_train = False # 是否对抗训练
    args.is_retrain = False # 是否重新训练对抗模型（否则从本地下载）  前提是is_adv_train为True

#######################################
#训练模型或者从本地下载训练好的模型
    if args.is_train == True:
        print("cnn is training......")
        train(cnn)
        print("finish training !")

    else:
        # load model
        print("download cnn from ./saved_model/cnn.pickle")
        f = open(r'./saved_model/cnn.pickle', 'rb')
        cnn = pickle.load(f)
        f.close()
        print("finish downloading !")
        # F1,correct = test(cnn)
        # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        # print("=" * 5, "test F1 is ", F1, "=" * 5, "test accuracy is ", correct)
    if args.is_test == True:
        test(cnn)
    if args.is_adv_train == True:
        print("对抗训练中......")
        if args.is_retrain == False:
            print("download retrained cnn  from ./saved_model/retrain_cnn.pickle")
            f = open(r'./saved_model/retrain_cnn.pickle', 'rb')
            cnn = pickle.load(f)
            f.close()
            print("finish downloading !")
        else:
            print("retrained cnn is retraining......")
            cnn = retrain(cnn)
            print("finish retraining !")
    epsfenzi_list = [1.,10.,20.,50.]
    for index in epsfenzi_list:
        args.epsfenzi = index
        args.eps = args.epsfenzi / 255
        ans = 0.
        if args.is_attack == True:
            print("攻击中...")
            print("eps: ",args.epsfenzi,r"/255")
            attack_success = 0.
            numlist = range(1,500,1)#[random.randint(0,499) for _ in range(100)]
            for i in numlist:
                sample_show_idx = i
                input_image = test_x[sample_show_idx]
                input_image = torch.unsqueeze(input_image,0)
                image_label = test_y_tensor[sample_show_idx]
                image_label = torch.unsqueeze(image_label, 0)
                if fgsm_type == "randomed":
                    loss_func = nn.CrossEntropyLoss()
                    attack_success += attack_fgsm(input_image, image_label, cnn, loss_func, args.eps)
                elif fgsm_type == "targeted":
                    target_label = (image_label+1)%10
                    attack_success += targeted_fgsm(input_image, image_label, target_label, cnn, args.eps)

            print("attack type is {}. Attack Successfully Rate(ASR) is: %.3f".format(fgsm_type) %(attack_success/len(numlist)))