'''
functions_pre.py用于写论文，会返回很多中间变量，比如说STN的参数，效果图，1D注意力机制的效果图等，写论文用
function.py 用于训练，只返回loss和output，训练用
'''
from math import nan
import os
import numpy as np
from PIL import Image
from torch import cuda
# from torch._C import device, long
from torch.functional import Tensor
from torch.nn.modules import loss
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
from time import time
import re 
## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)
def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()
def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()
def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()
## ---------------------- For STN module ---------------------- ##



#27-168行没用，舍弃的方案，边缘检测手动聚焦的方法
def draw_rect(src):  # 1/4 rectangle
    t=80
    binary = cv2.Canny(src, t, t * 2,L2gradient=True)
    k = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, k)
    contours_x, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours=contours_x[-1]
    for c in range(len(contours)):
        rect = cv2.minAreaRect(contours[c])  # 外接矩形
        cx, cy = rect[0]  # 中心点
        # M=cv2.getRotationMatrix2D(())
        box = cv2.boxPoints(rect)  # 左下左上右上右下
        left_top = np.int32(box[1][0]), np.int32(box[1][1])
        left_bottom = np.int32((box[0][0] + box[1][0]) / 2), np.int32((box[0][1] + box[1][1]) / 2)
        right_top = np.int32((box[1][0] + box[2][0]) / 2), np.int32((box[1][1] + box[2][1]) / 2)
        right_bottom = np.int32(cx), np.int32(cy)
        rotation = rect[2]
        if rotation < 0:
            T_left_top = left_top[0], right_top[1]
            T_right_bottom = right_bottom[0], left_bottom[1]
        else:
            T_left_top = left_bottom[0], left_top[1]
            T_right_bottom = right_top[0], right_bottom[1]
        height=T_right_bottom[1]-T_left_top[1]
        weight=T_right_bottom[0]-T_left_top[0]
        crop=src[T_left_top[1]:T_left_top[1]+height,T_left_top[0]:T_left_top[0]+weight]

        cv2.destroyAllWindows()
        return crop


def get_min(x):
    return min(x)
def draw_left_rect(src):
    t = 80
    #src=cv2.imread(image)
    binary = cv2.Canny(src, t, t * 2, L2gradient=True)
    s = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, s)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = contours_x[-1]
    #if len(contours) == 2:
        #del contours[0]

    for c in range(len(contours)):
        area=cv2.contourArea(contours[c])
        if area<=5000:
            continue
        rect = cv2.minAreaRect(contours[c])  # 外接矩形
        cx, cy = rect[0]  # 中心点
        # M=cv2.getRotationMatrix2D(())
        box = cv2.boxPoints(rect)  # random squence
        rotation = rect[2]
        left_bottom, left_top, right_top, right_bottom = (box[0][0], box[0][1]), (box[1][0], box[1][1]), \
                         (box[2][0], box[2][1]), (box[3][0], box[3][1])
        list = []
        list.append(left_bottom)
        list.append(left_top)
        list.append(right_top)
        list.append(right_bottom)
        #print(list)
        x=[left_bottom[0],left_top[0],right_top[0],right_bottom[0]]
        y=[left_bottom[1],left_top[1],right_top[1],right_bottom[1]]
        num=x[0]
        if num not in x[1:]:
            new_x=sorted(x)
            min=new_x[0]
            min_2=new_x[1]
            idx=x.index(min)
            idx2=x.index(min_2)
            if y[idx]<y[idx2]:
                #print('left type')
                x=x[idx:]+x[:idx]
                y=y[idx:]+y[:idx]
                left_top=x[0],y[0]
                right_top=x[1],y[1]
                right_bottom=x[2],y[2]
                left_bottom=x[3],y[3]
            else:
                #print('right type')
                x=x[idx:]+x[:idx]
                y=y[idx:]+y[:idx]
                left_bottom=x[0],y[0]
                left_top=x[1],y[1]
                right_top=x[2],y[2]
                right_bottom=x[3],y[3]

        else:
            #print('rect')
            min_w=get_min(x)
            max_w=max(x)
            min_y=get_min(y)
            max_y=max(y)
            left_top=min_w,min_y
            left_bottom=min_w,max_y
            right_top=max_w,min_y
            right_bottom=max_w,max_y

        #print(left_top,right_top,right_bottom,left_bottom)


        if left_top[1]>right_top[1] and abs(left_top[1]-right_top[1])>0.01:
            state='LEFT'
            k=(left_top[1]-right_top[1])/(right_top[0]-left_top[0])
            delte_x=cx-left_top[0]
            delta_y=delte_x*k
            T_y=left_top[1]-delta_y
            T_left_top=np.int32(left_top[0]),np.int32(T_y)
            T_right_bottom=np.int32(cx),np.int32(left_bottom[1])

        if abs(left_top[1]-right_top[1])<0.001:
            state='BALANCE'
            T_left_top=np.int32(left_top[0]),np.int32(left_top[1])
            T_right_bottom=np.int32(right_bottom[0]),np.int32(right_bottom[0])
            k=0

        if left_top[1]<right_top[1] and abs(left_top[1]-right_top[1])>0.01:
            state='RIGHT'
            k=(right_top[1]-left_top[1])/(right_top[0]-left_top[0])
            delta_x=right_top[0]-cx
            delta_y=delta_x*k
            T_y=right_top[1]-delta_y
            T_left_top=np.int32(left_bottom[0]),np.int32(T_y)
            T_right_bottom=np.int32(cx),np.int32(right_bottom[1])

       # print(state,k,T_left_top,T_right_bottom)
        list.append(k)
        height=T_right_bottom[1]-T_left_top[1]
        weight=T_right_bottom[0]-T_left_top[0]
        # ration=2
        # if height/weight >=ration:
        #if T_left_top[1]>220:
            #print(rotation,left_top, right_top, right_bottom, left_bottom,len(contours))
        if state=='LEFT' :
            crop=src[T_left_top[1]:T_left_top[1]+height,T_left_top[0]-weight:T_left_top[0]+weight]
        if state=='RIGHT':
            crop = src[T_left_top[1]:T_left_top[1] + height, T_left_top[0]-weight:T_left_top[0]+weight]
        if state=='BALANCE':
            crop = src[T_left_top[1]:T_left_top[1] + int(0.5*height), T_left_top[0]-2*int(0.5*weight):T_left_top[0] + 2*int(0.5*weight)]
        return crop
## ---------------------- End of STN module ---------------------- ##
##

## ---------------------- Dataloaders ---------------------- ##
class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, frames, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        X_label = []
        # 均匀取帧
        frames =self.frames
        length = len(os.listdir(os.path.join(path,selected_folder)))
        margin = int(length/frames)
        for i in range(frames):
            # img_path = os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin))
            for img_patient in os.listdir(os.path.join(path,selected_folder)):
                # print( img_patient[:4])
                if '{:04d}'.format(i+margin) == img_patient[:4]:
                    each_label = img_patient[-5:-4]
                    X_label.append(int(each_label))
                    image = Image.open(os.path.join(path, selected_folder, img_patient))
                    if use_transform is not None:
                        image = use_transform(image)
                    X.append(image)
        #  师兄版本
        '''
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, '{:04d}.jpg'.format(i)))
            #print(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))
            #image=cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            #image = draw_left_rect(image)
            #image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if use_transform is not None:
                image = use_transform(image)
            X.append(image)
        '''
        X = torch.stack(X, dim=0)
        X_label=torch.tensor(X_label,dtype=torch.float)
        # Z = Image.open(os.path.join(path,selected_folder,'area.jpg'))
        # Z = np.array(Z)
        return X , X_label #,Z

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data 删去z
        X ,X_label= self.read_images(self.data_path, folder, self.transform)     # (input) spatial images
        y = torch.LongTensor([int(self.labels[index])])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # Z = torch.from_numpy(np.transpose(Z,[2,0,1]).astype(float)/255)
        # Z = Z.type(torch.FloatTensor)
        return X, y, X_label

## ---------------------- end of Dataloaders ---------------------- ##


## ---------------------- Dataloaders ---------------------- ##
class gccs_branch_Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, frames, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        X_label = []
        # 均匀取帧
        frames =self.frames
        length = len(os.listdir(os.path.join(path,selected_folder)))
        # margin = int(length/frames)
        for i in range(frames):
            # img_path = os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin))
            for img_patient in os.listdir(os.path.join(path,selected_folder)):
                # print( img_patient[:4])
                if '{:04d}'.format(i) == img_patient[:4]:

                    each_label = img_patient[-5:-4]
                    if each_label == '2':
                        each_label =='0'
                    X_label.append(int(each_label))
                    image = Image.open(os.path.join(path, selected_folder, img_patient))
                    if use_transform is not None:
                        image = use_transform(image)
                    X.append(image)
        #  师兄版本
        '''
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, '{:04d}.jpg'.format(i)))
            #print(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))
            #image=cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            #image = draw_left_rect(image)
            #image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if use_transform is not None:
                image = use_transform(image)
            X.append(image)
        '''
        X = torch.stack(X, dim=0)
        X_label=torch.tensor(X_label,dtype=torch.float)
        # Z = Image.open(os.path.join(path,selected_folder,'area.jpg'))
        # Z = np.array(Z)
        return X , X_label #,Z

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data 删去z
        X ,X_label= self.read_images(self.data_path, folder, self.transform)     # (input) spatial images
        y = torch.LongTensor([int(self.labels[index])])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # Z = torch.from_numpy(np.transpose(Z,[2,0,1]).astype(float)/255)
        # Z = Z.type(torch.FloatTensor)
        return X, y, X_label

## ---------------------- end of Dataloaders ---------------------- ##






## -------------------- (reload) model prediction ---------------------- ##
def CRNN_final_prediction(model, device, loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, Z,y) in enumerate(tqdm(loader)):
            # distribute data to device
            Z = Z.to(device)
            X = X.to(device)
            output = rnn_decoder(cnn_encoder(X),Z)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred

## -------------------- end of model prediction ---------------------- ##





## ------------------------ CRNN module ---------------------- ##
# 2D CNN encoder using ResNet-152 pretrained
class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.dropout=nn.Dropout(drop_p)

        #resnet = models.resnet152(pretrained=True)
        # Mobilenetv2方案已经舍弃
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        # 新建2D分支
        self.classifier = nn.Sequential(
        nn.Linear(CNN_embed_dim,100),
        nn.ReLU(),
        nn.Linear(100, 1),
        )
        #######
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

## ---------------------- STN module ---------------------- ##
        self.localization=nn.Sequential(
            nn.Conv2d(3,8,kernel_size=3),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(8,16,kernel_size=3),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(16,32,kernel_size=3),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5),
            # nn.MaxPool2d(2, stride=2),
            # nn.ReLU(True)
        )
        self.fc_loc=nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(True),
            nn.Linear(16,6)
        )
        #注意注意！！！！STN依赖参数初始化
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))  #STN module的参数初始化非常条件，0初始化或者选择高斯初始化，方差极低
## ---------------------- ECA module ---------------------- ##
        self.eca1=nn.Sequential(
            nn.Conv1d(1,1,kernel_size=3,padding=1,bias=False),
            nn.Sigmoid()
        )
        self.eca2=nn.Sequential(
            nn.Conv1d(1,1,kernel_size=3,padding=1,bias=False),
            nn.Sigmoid()
        )
        self.eca3=nn.Sequential(
            nn.Conv1d(1,1,kernel_size=3,padding=1,bias=False),
            nn.Sigmoid()
        )

## ----------------------End of ECA module ---------------------- ##
    def stn(self,x):
        xs=self.localization(x)
        xs=xs.view(-1,32)
        theta=self.fc_loc(xs)
        theta=theta.view(-1,2,3)
        grid=F.affine_grid(theta,x.size())   #构建STN自适应仿射变换
        x=F.grid_sample(x,grid)
        return x


    def forward(self, x_3d):
        cnn_embed_seq = []
        branch_predict =[]
        #整体流程
        #编码器
        #29*224*224的图像，每次取一帧--->stn--->ResNet18--->3个FC，嵌套ECa模块--->29*512

        #解码器
        #29*×512的结果，第一个LSTM迭代29次，第二个LSTM迭代29次，取最后一个LSTM的输出，1*512
        #area_curve经过5次conv/poll，得到1*64
        #特征融合1*512,1*64得到1*576，
        #1*576经过2个FC出结果




        for t in range(x_3d.size(1)):
            x=x_3d[:, t, :, :, :]

            x=self.stn(x)
            # ResNet CNN
            # with torch.no_grad():
            #     x = self.resnet(x_3d[:, t, :, :, :])
            #     x = x.view(x.size(0), -1)       #原本不希望更新ResNet-18参数，但是发现更新更好，所以舍弃

            x=self.resnet(x)       #50 512 1 1
            
            x=x.view(x.size(0), -1)   #50 512
            

            y=x.unsqueeze(1)     #第一个ECA
            y=self.eca1(y)
            y=y.squeeze(1)
            x=x*y

            # FC layers
            x = self.bn1(self.fc1(x)) #全连接
            x = F.relu(x)

            y2=x.unsqueeze(1)  #再ECA
            y2=self.eca2(y2)
            y2=y2.squeeze(1)
            x=x*y2

            x = self.bn2(self.fc2(x)) #全连接
            x = F.relu(x)

            y3=x.unsqueeze(1) #第三个ECA
            y3=self.eca3(y3)
            y3=y3.squeeze(1)
            x=x*y3


            x=self.dropout(x)
            x = self.fc3(x)
            cnn_embed_seq.append(x)
            
            x_predict= self.classifier(x)
            branch_predict.append(x_predict)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  #stack到29个，（1*512），变成29×512
        # cnn_embed_seq: shape=(batch, time_step, input_size)
        branch_predict = torch.stack(branch_predict,dim=0).transpose_(0, 1).squeeze() 
        
        return cnn_embed_seq,branch_predict


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=2,area=32):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.area=area
        self.drop_out = nn.Dropout(drop_p)

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        #以上为LSTM的，一下为处理Area_curve的网络
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        # se
        self.fc_z = nn.Linear(4800,256)

        self.fc_z2 = nn.Linear(256,self.area)
        ##above for area_process

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Sequential(
        nn.Linear(self.h_FC_dim, self.num_classes),
        nn.Sigmoid()
        )
    def forward(self, x_RNN):
        
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        # x_z = self.conv1(Z)
        # x_z = self.conv2(x_z)
        # x_z = self.drop_out(x_z)

        # x_z = self.conv3(x_z)
        # x_z = self.conv4(x_z)
        # x_z = self.drop_out(x_z)

        # x_z = self.conv5(x_z)
        # x_z = x_z.view(-1,4800)
        # x_z = self.fc_z(x_z)
        # x_z = self.fc_z2(x_z)

        x=RNN_out[:, -1, :]#选择LSTM的最后一个输出，作为预测
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x=self.drop_out(x)
        x = self.fc2(x)
      
        return x

## ---------------------- end of CRNN module ---------------------- ##

##   ----------------------weightlosss---------------------------------## 

class weightBCELoss(nn.Module):
    def __init__(self,weight):
        super(weightBCELoss,self).__init__()
        self.weight =weight
    
    def com(x,y):
        return -(y * torch.log(x) + (1-y)*torch.log(1-x))

    def forward(self,input:Tensor,target:Tensor,device):
        loss = torch.tensor(0,dtype=torch.float32).to(device=device)
        for i in range(input.size(0)):
           sum= torch.tensor(0,dtype=torch.float32).to(device=device)
           for j in range(input.size(1)):
                # print(target[i,j])
                num=( -(target[i,j]* torch.log(input[i,j]) + (1-target[i,j])*torch.log(1-input[i,j])))
                if num.isnan():
                    num =torch.tensor(0,dtype=torch.float32).to(device=device)
                sum+=num
           loss += (self.weight[i]*sum)
        return loss/input.size(0)


## ---------------------- end of CRNN module ---------------------- ##