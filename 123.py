# coding=utf-8
import os
from math import sqrt
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import _utils
import cv2
#from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import torchvision.models
from tqdm import tqdm, trange
import torch.nn as nn
print('mmmmmmmmmmmmmmmmmmmmmm',torch.cuda.is_available())


# 我们读取图片的根目录， 在根目录下有所有图片的txt文件， 拿到txt文件后， 先读取txt文件， 之后遍历txt文件中的每一行， 首先去除掉尾部的换行符， 在以空格切分，前半部分是图片名称， 后半部分是图片标签， 当图片名称和根目录结合，就得到了我们的图片路径
class MyDataset(Dataset):
    def __init__(self, img_path, transform=None):
        super(MyDataset, self).__init__()
        self.root = img_path
        f = open(self.root, 'r')
        data = f.readlines()

        imgs = []
        labels = []
        labels2 = []
        for line in data:
            line = line.rstrip()
            word = line.split()
            imgs.append(os.path.join(word[0]))
            labels.append(word[2])
            labels2.append(word[3])
        self.img = imgs
        self.label = labels
        self.label2 = labels2
        self.transform = transform


        # labels2 = [int(x) for x in labels2]  # 将字符串类型转换为整数类型
        # labels2_onehot = torch.nn.functional.one_hot(torch.tensor(labels2), num_classes=3)
        # self.labels2 = labels2.float()
        #print(self.img)
        # print(self.label)y

    def __len__(self):
        return len(self.label)
        return len(self.img)
        return len(self.label2)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]
        label2 = self.label2[item]

        # print(img)
        # img = Image.open(img).convert('RGBA')

        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        # 此时img是PIL.Image类型   label是str类型
        # print(img)
        img_array = np.array(img)

        # 获取 RGB 三通道

        img_rgb = img_array[:, :, [2, 1, 0]].astype(np.float32) / 255.0
        # print(img_rgb)
        # 更改RGB三通道的顺序为BGR并归一化
        # print(img_rgb)
        # 获取 DEM 第四通道
        img_dem = img_array[:, :, 3]
        img_dem = (img_dem - img_dem.min()) / (img_dem.max() - img_dem.min())
        # print(img_dem)

        # 合并 RGB 三通道和 DEM 第四通道
        img_array = np.dstack((img_rgb, img_dem))

        # print(img_array)
        # 将归一化后的 numpy 数组转换为 PIL.Image 对象
        img = Image.fromarray((img_array * 255.0).astype('float32').astype('uint8')).convert('RGBA')

        if transforms is not None:
            img = self.transform(img)

        label = np.array(label).astype(np.float32)
        label = torch.from_numpy(label)
        label2 =np.array(label2).astype(np.float32)
        return img, label,label2





        #
        # if transforms is not None:
        #     img = self.transform(img)
        #     # print(img.max())
        #
        # label = np.array(label).astype(np.float32)
        # label = torch.from_numpy(label)
        # return img, label

root_train = r"C:\Users\J\Desktop\906RG\datapath\train1_list_1.txt"
root_val = r"C:\Users\J\Desktop\906RG\datapath\val_list_1.txt"
root_test = r"C:\Users\J\Desktop\906RG\datapath\test_list_1.txt"
#data_tf = transforms.Compose([transforms.ToTensor()])
#train_data = MyDataset(root_train, transform=data_tf)



#normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(), # 按0.5的概率水平翻转图片
     # 对图像四周各填充4个0像素，然后随机裁剪成32*32
    transforms.ToTensor(),
    # normalize,
    ])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    # normalize,
    ])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    # normalize,
    ])

train_dataset = MyDataset(root_train, transform=train_transform)
val_dataset = MyDataset(root_val, transform=val_transform)
test_dataset = MyDataset(root_test, transform=test_transform)

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=4, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=4, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# for batch, (x1, y1) in enumerate(train_dataloader):
#     xtrain, ytrain = x1.to(device), y1.to(device)
# for batch, (x2, y2) in enumerate(val_dataloader):
#     xval, yval = x2.to(device), y2.to(device)



# Hyper parameters 超参数
num_epochs = 200
# batch_size: int = 4
learning_rate = 0.00001



def main():

    # model = torchvision.models.resnet50(pretrained=True)
    # model = model.cuda()
    # model = model.to(device)
    # model.fc = nn.Linear(2048,1)

    model = torchvision.models.resnet50(pretrained=True)

    weight = model.conv1.weight
    new_weight = torch.nn.Parameter(torch.cat((weight, weight[:, :1, :, :]), dim=1))
    model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1.weight = new_weight

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




    #train the model
    total_step = len(train_dataloader)
    min_loss = 1000
    for epoch in range(num_epochs):
        LOSS = 0
        VALLOSS = 0
        rmse_sum=0
        n=0
        rmse_sum1 = 0
        n1 = 0
        model.train()
        for i, (images, labels, labels2) in enumerate(tqdm(train_dataloader)):
            images = images.to(device)

            labels = labels.to(torch.float32)
            labels2=labels2.to(device)
            labels = np.around(labels, decimals=2)
            labels = labels.to(device)
            # Forward pass
            model = model.to(device)
            outputs = model(images)
            outputs = outputs.to(torch.float32)

            labels2 = torch.Tensor(labels2).unsqueeze(1)
            # print(labels2.shape)
            # print(outputs.shape)
            outputs = torch.cat((outputs, labels2.to(device)), dim=0)

            loss = criterion(outputs, labels)
            LOSS += loss.item()
            print(outputs)
            print(labels)

            #RMSE
            # rmse_sum =  rmse_sum.to_bytes(device)
            rmse_sum += torch.sqrt(((model(images.to(device)) - labels) ** 2).sum()).cpu().item()
            #R2
            #
            # print(outputs)
            # print(labels)
            # print(loss)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

        train_loss = LOSS / len(train_dataloader)
        train_rmse = rmse_sum / sqrt((len(train_dataloader)*4-2))
        #验证
        print(len(train_dataloader))
        print(n)
        with torch.no_grad():
            for (images, labels ,labels2) in val_dataloader:
                images = images.to(device)

                labels = labels.to(torch.float32)
                labels = labels.to(device)
                labels2 = labels2.to(device)
                labels2 = torch.Tensor(labels2).unsqueeze(1)

                outputs = torch.cat((outputs, labels2.to(device)), dim=0)


                valloss = criterion(outputs, labels)
                VALLOSS += valloss.item()
                rmse_sum1 += torch.sqrt(((model(images.to(device)) - labels) ** 2).sum()).cpu().item()
            Valloss = VALLOSS / len(val_dataloader)
            val_rmse = rmse_sum1 / sqrt((len(train_dataloader)*4-2))
            if Valloss < min_loss:
                min_loss = Valloss
                print("save model")
                # 保存模型语句
                torch.save(model.state_dict(), "model.pth")
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f},train_rmse: {:.4f},valLoss: {:.4f},valrmse: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, total_step, train_loss, train_rmse, Valloss, val_rmse))
        Loss0 = np.array(train_loss)
        np.save(r'C:\Users\J\Desktop\loss\train/epoch_{}'.format(epoch), Loss0)
        Loss1 = np.array(Valloss)
        np.save(r'C:\Users\J\Desktop\loss\test/epoch_{}'.format(epoch), Loss1)

    #Test the model



    #Save the model checkpoint
    #torch.save(model.state_dict(), 'model2.ckpt')


    #test
    # TESTLOSS = 0
    #
    # model.load_state_dict(torch.load('model.pth'))
    # model.eval()
    # with torch.no_grad():
    #     for images, labels in test_dataloader:
    #         images = images.to(device)
    #
    #         labels = labels.to(torch.float32)
    #         labels = labels.to(device)
    #
    #         outputs = model(images)
    #         #outputs = outputs.squeeze()
    #         #outputs = torch.round(outputs)
    #         testloss = criterion(outputs, labels.view([-1, 1]))
    #         #testloss += loss.item()
    #         #with open(r"C:\Users\J\Desktop\1.pt",'ab') as f:
    #             #torch.save(outputs,f)
    #         #with open(r"C:\Users\J\Desktop\2.pt",'ab') as f:
    #             #torch.save(labels, f)
    #         print(outputs)
    #         print(labels)
    #     #Testloss=testloss/len(test_dataloader)
if __name__ == '__main__':
    main()
