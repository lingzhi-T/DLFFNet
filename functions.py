'''
functions_pre.py用于写论文，会返回很多中间变量，比如说STN的参数，效果图，1D注意力机制的效果图等，写论文用
function.py 用于训练，只返回loss和output，训练用
'''
import os
from cv2 import imshow
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
from time import time

## __________________________tensor2image___________##
import torch
from PIL import Image
import matplotlib.pyplot as plt
 
# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])  
 
unloader = transforms.ToPILImage()
def imshowimg(tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

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


class baseline_Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, mask_path,folders, labels, frames, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames
        self.mask_path = mask_path

    def __len__(self):
        return len(self.folders)
   
    def read_images(self, path,mask_path, selected_folder, use_transform):
        X = []
        # 均匀取帧
        # frames =self.frames
        # length = len(os.listdir(os.path.join(path,selected_folder)))
        # margin = int(length/frames)
        # for i in range(frames):
        #     image = Image.open(os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin)))
        #     if use_transform is not None:
        #         image = use_transform(image)
        #     X.append(image)
        #  

        ##   截断 之后
        frames =self.frames
        length = len(os.listdir(os.path.join(path,selected_folder)))
        # margin = int(length/frames)
        for i in range(length):
            # img_path = os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin))
            for img_patient in os.listdir(os.path.join(path,selected_folder)):
                if '{:04d}'.format(i) == img_patient[:4]:
                    # each_label = img_patient[-5:-4]
                    # X_label.append(int(each_label))
                    mask = Image.open(os.path.join(mask_path,selected_folder,img_patient))
                    image = Image.open(os.path.join(path, selected_folder, img_patient))
                    if use_transform is not None:
                        image = use_transform(image)
                        mask = use_transform(mask)
                        # image_mask = image*mask
                        # imshowimg(image_mask)

                    X.append(image)
         
         # 均匀取帧
        # frames =self.frames
        # length = len(os.listdir(os.path.join(path,selected_folder)))
        # margin = int(length/frames)
        # for i in range(frames):
        #     # img_path = os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin))
        #     for img_patient in os.listdir(os.path.join(path,selected_folder)):
        #         if '{:04d}'.format(i+margin) == img_patient[:4]:
        #             each_label = img_patient[-5:-4]
                   
        #             image = Image.open(os.path.join(path, selected_folder, img_patient))
        #             if use_transform is not None:
        #                 image = use_transform(image)
        #             X.append(image)
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
        # Z = Image.open(os.path.join(path,selected_folder,'area.jpg'))
        # Z = np.array(Z)
        return X #,Z

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data 删去z
        X = self.read_images(self.data_path,self.mask_path, folder, self.transform)     # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # Z = torch.from_numpy(np.transpose(Z,[2,0,1]).astype(float)/255)
        # Z = Z.type(torch.FloatTensor)
        return X, y

## ---------------------- end of Dataloaders ---------------------- ##

class img_baseline_Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path,folders, labels, frames, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames
        

    def __len__(self):
        return len(self.folders)
   
    def read_images(self, path, selected_folder, use_transform):
        X = []
        # 均匀取帧
        # frames =self.frames
        # length = len(os.listdir(os.path.join(path,selected_folder)))
        # margin = int(length/frames)
        # for i in range(frames):
        #     image = Image.open(os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin)))
        #     if use_transform is not None:
        #         image = use_transform(image)
        #     X.append(image)
        #  

        ##   截断 之后
        frames =self.frames
        length = len(os.listdir(os.path.join(path,selected_folder)))
        # margin = int(length/frames)
        for i in range(length):
            # img_path = os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin))
            for img_patient in os.listdir(os.path.join(path,selected_folder)):
                if '{:04d}'.format(i) == img_patient[:4]:
                    # each_label = img_patient[-5:-4]
                    # X_label.append(int(each_label))
                   
                    image = Image.open(os.path.join(path, selected_folder, img_patient))
                    if use_transform is not None:
                        image = use_transform(image)
                
                        # image_mask = image*mask
                        # imshowimg(image_mask)

                    X.append(image)
         
         # 均匀取帧
        # frames =self.frames
        # length = len(os.listdir(os.path.join(path,selected_folder)))
        # margin = int(length/frames)
        # for i in range(frames):
        #     # img_path = os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin))
        #     for img_patient in os.listdir(os.path.join(path,selected_folder)):
        #         if '{:04d}'.format(i+margin) == img_patient[:4]:
        #             each_label = img_patient[-5:-4]
                   
        #             image = Image.open(os.path.join(path, selected_folder, img_patient))
        #             if use_transform is not None:
        #                 image = use_transform(image)
        #             X.append(image)
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
        # Z = Image.open(os.path.join(path,selected_folder,'area.jpg'))
        # Z = np.array(Z)
        return X #,Z

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data 删去z
        X = self.read_images(self.data_path, folder, self.transform)     # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # Z = torch.from_numpy(np.transpose(Z,[2,0,1]).astype(float)/255)
        # Z = Z.type(torch.FloatTensor)
        return X, y

## ---------------------- end of Dataloaders ---------------------- ##

## ---------------------- Dataloaders ---------------------- ##
class mask_Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, mask_path,folders, labels, frames, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames
        self.mask_path = mask_path

    def __len__(self):
        return len(self.folders)
   
    def read_images(self, path,mask_path, selected_folder, use_transform):
        X = []
        # 均匀取帧
        # frames =self.frames
        # length = len(os.listdir(os.path.join(path,selected_folder)))
        # margin = int(length/frames)
        # for i in range(frames):
        #     image = Image.open(os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin)))
        #     if use_transform is not None:
        #         image = use_transform(image)
        #     X.append(image)
        #  

        ##   截断 之后
        frames =self.frames
        length = len(os.listdir(os.path.join(path,selected_folder)))
        # margin = int(length/frames)
        for i in range(length):
            # img_path = os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin))
            for img_patient in os.listdir(os.path.join(path,selected_folder)):
                if '{:04d}'.format(i) == img_patient[:4]:
                    # each_label = img_patient[-5:-4]
                    # X_label.append(int(each_label))
                    mask = Image.open(os.path.join(mask_path,selected_folder,img_patient))
                    image = Image.open(os.path.join(path, selected_folder, img_patient))
                    if use_transform is not None:
                        image = use_transform(image)
                        mask = use_transform(mask)
                        image_mask = image*mask
                        # imshowimg(image_mask)

                    X.append(image_mask)
         
         # 均匀取帧
        # frames =self.frames
        # length = len(os.listdir(os.path.join(path,selected_folder)))
        # margin = int(length/frames)
        # for i in range(frames):
        #     # img_path = os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin))
        #     for img_patient in os.listdir(os.path.join(path,selected_folder)):
        #         if '{:04d}'.format(i+margin) == img_patient[:4]:
        #             each_label = img_patient[-5:-4]
                   
        #             image = Image.open(os.path.join(path, selected_folder, img_patient))
        #             if use_transform is not None:
        #                 image = use_transform(image)
        #             X.append(image)
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
        # Z = Image.open(os.path.join(path,selected_folder,'area.jpg'))
        # Z = np.array(Z)
        return X #,Z

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data 删去z
        X = self.read_images(self.data_path,self.mask_path, folder, self.transform)     # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # Z = torch.from_numpy(np.transpose(Z,[2,0,1]).astype(float)/255)
        # Z = Z.type(torch.FloatTensor)
        return X, y

## ---------------------- end of Dataloaders ---------------------- ##

## ---------------------- 0401Dataloaders ---------------------- ##
class Dataset_CRNN_mask(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, mask_path,folders, labels, frames, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames
        self.mask_path = mask_path

    def __len__(self):
        return len(self.folders)
   
    def read_images(self, path,mask_path, selected_folder, use_transform):
        X = []
        X_mask=[]
        # 均匀取帧
        # frames =self.frames
        # length = len(os.listdir(os.path.join(path,selected_folder)))
        # margin = int(length/frames)
        # for i in range(frames):
        #     image = Image.open(os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin)))
        #     if use_transform is not None:
        #         image = use_transform(image)
        #     X.append(image)
        #  

        ##   截断 之后
        frames =self.frames
        length = len(os.listdir(os.path.join(path,selected_folder)))
        # margin = int(length/frames)
        for i in range(length):
            # img_path = os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin))
            for img_patient in os.listdir(os.path.join(path,selected_folder)):
                if '{:04d}'.format(i) == img_patient[:4]:
                    # each_label = img_patient[-5:-4]
                    # X_label.append(int(each_label))
                    mask = Image.open(os.path.join(mask_path,selected_folder,img_patient))
                    # mask = Image.open('/home/tlz/labelme-master/examples/semantic_segmentation/data_dataset_voc/SegmentationClassPNG/0000.png')
                    image = Image.open(os.path.join(path, selected_folder, img_patient))
                    if use_transform is not None:
                        image = use_transform[1](image)
                        mask = use_transform[0](mask)
                        # image_mask = image*mask
                        # imshowimg(image_mask)

                    X.append(image)
                    X_mask.append(mask)
         
        X = torch.stack(X, dim=0)
        X_mask=torch.stack(X_mask,dim=0)
        # Z = Image.open(os.path.join(path,selected_folder,'area.jpg'))
        # Z = np.array(Z)
        return X ,  X_mask

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data 删去z
        X ,X_mask= self.read_images(self.data_path,self.mask_path, folder, self.transform)     # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # Z = torch.from_numpy(np.transpose(Z,[2,0,1]).astype(float)/255)
        # Z = Z.type(torch.FloatTensor)
        return X, y,X_mask

## ---------------------- end of Dataloaders ---------------------- ##

## ---------------------- start of optiocal flow dataset 0512  ---------------------- ##
class Dataset_img_optical(data.Dataset):
    # 0512 光流路径来源
    def __init__(self, data_path, flow_path,mask_path,folders, labels, frames, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames
        self.mask_path = mask_path
        self.flow_path = flow_path

    def __len__(self):
        return len(self.folders)
   
    def read_images(self, path,flow_path,mask_path, selected_folder, use_transform):
        X = []
        X_mask=[]
        X_flow =[]
        frames =self.frames
        length = len(os.listdir(os.path.join(path,selected_folder)))
        # margin = int(length/frames)
        for i in range(length):
            # img_path = os.path.join(path, selected_folder, '{:04d}.jpg'.format(i+margin))
            for img_patient in os.listdir(os.path.join(path,selected_folder)):

                if '{:04d}'.format(i) == img_patient[:4]:
                    # each_label = img_patient[-5:-4]
                    # X_label.append(int(each_label))
                    mask = Image.open(os.path.join(mask_path,selected_folder,img_patient))
                    image = Image.open(os.path.join(path, selected_folder, img_patient))
                    if i > 0:
                        x_flow = Image.open(os.path.join(flow_path, selected_folder, 'x_'+ '{:05d}'.format(i)+'.jpg'))
                        x_flow =use_transform[0](x_flow)
                        y_flow = Image.open(os.path.join(flow_path, selected_folder, 'y_'+ '{:05d}'.format(i)+'.jpg'))
                        y_flow =use_transform[0](y_flow)
                        X_flow.append(x_flow)
                        X_flow.append(y_flow)
                    if use_transform is not None:
                        image = use_transform[1](image)
                        mask = use_transform[0](mask)
                        # image_mask = image*mask
                        # imshowimg(image_mask)

                    X.append(image)
                    X_mask.append(mask)
                    
         

   
        X = torch.stack(X, dim=0)
        X_mask=torch.stack(X_mask,dim=0)
        X_flow = torch.stack(X_flow,dim=0)
        # Z = Image.open(os.path.join(path,selected_folder,'area.jpg'))
        # Z = np.array(Z)
        return X ,  X_mask,X_flow

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data 删去z
        X ,X_mask,X_flow= self.read_images(self.data_path,self.flow_path,self.mask_path, folder, self.transform)     # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                  # (labels) LongTensor are for int64 instead of FloatTensor

        # Z = torch.from_numpy(np.transpose(Z,[2,0,1]).astype(float)/255)
        # Z = Z.type(torch.FloatTensor)
        return X, y,X_mask,X_flow









## ---------------------- end of Dataloaders ---------------------- ##

## -------------------- (reload) model prediction ---------------------- ##
def CRNN_final_prediction(model, device, loader):
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            # Z = Z.to(device)
            X = X.to(device)
            # output = rnn_decoder(cnn_encoder(X))
            output = cnn_encoder(X)
            
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred

## -------------------- end of model prediction ---------------------- ##
from prettytable import PrettyTable
from pretty_confusion_matrix import pp_matrix
import numpy as np
import pandas as pd
class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self,result_f):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        cmap = 'Blues'
        df_cm = pd.DataFrame(self.matrix, index=['score 0','score 1','score 2'], columns=['score 0','score 1','score 2'])
        pp_matrix(df_cm, cmap=cmap,title="Backbone")
        acc = sum_TP / np.sum(self.matrix)
        result_f.write(f"\nthe model accuracy is{acc}")
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity","f1_score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            f1_score = 1./(1/(Precision+0.00001) + 1/(Recall+0.00001))
            table.add_row([self.labels[i], Precision, Recall, Specificity,f1_score])
        print(table)
        result_f.write("\n")
        result_f.write(str(table))
        return acc

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
## -------------------- (reload) model prediction ---------------------- ##
def final_prediction(model, device, loader,result_f):
    
    confusion=ConfusionMatrix(num_classes=3,labels=[0,1,2])
    cnn_encoder = model
    cnn_encoder.eval()
    # rnn_decoder.eval()
    all_out =[]
    all_y_pred = []
    all_y =[]
    with torch.no_grad():
        for batch_idx, (X, y,X_mask) in enumerate(tqdm(loader)):
            # distribute data to device
            # Z = Z.to(device)
            X = X.to(device)
            print('x_mask',X_mask.shape)
            X_mask = X_mask.to(device)
            # print(X_mask.shape)
            # output = rnn_decoder(cnn_encoder(X))
            output = cnn_encoder(X,X_mask)
            output = nn.Softmax(dim=1)(output)
            all_out+=output
            all_y+=y
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.numpy().tolist())
            confusion.update(np.array(y_pred.to("cpu")),y.to("cpu").numpy())
    acc_best_now=confusion.summary(result_f)
    # all_out = torch.stack(all_out,dim=0).to("cpu").view(-1,3)
    all_out = torch.stack(all_out,dim=0).to("cpu")
    all_y = torch.stack(all_y,dim=0).to("cpu")
    return all_y_pred,acc_best_now,all_out,all_y

## -------------------- end of model prediction ---------------------- ##
import matplotlib.pyplot as plt
import scipy.misc
import imageio

def show_feature_map(feature_map,store_path_name):#feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
                                                                         # feature_map[2].shape     out of bounds
    feature_map = feature_map.squeeze(0)#压缩成torch.Size([64, 55, 55])
    
    #以下4行，通过双线性插值的方式改变保存图像的大小
    feature_map =feature_map.view(1,feature_map.shape[0],feature_map.shape[1],feature_map.shape[2])#(1,64,55,55)
    upsample = torch.nn.UpsamplingBilinear2d(size=(256,256))#这里进行调整大小
    feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1],feature_map.shape[2],feature_map.shape[3])
    
    feature_map_num = feature_map.shape[0]#返回通道数
    row_num = int(np.ceil(np.sqrt(feature_map_num)))#8
    plt.figure()
    for index in range(1, feature_map_num + 1):#通过遍历的方式，将64个通道的tensor拿出

        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index - 1].cpu().detach().numpy()*255, cmap='gray')#feature_map[0].shape=torch.Size([55, 55])
        #将上行代码替换成，可显示彩色 plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
        plt.close()
        imageio.imwrite( store_path_name+str(index) + ".png", feature_map[index - 1].cpu().detach())
        
    plt.show()
    plt.close()  # 就是这里 一定要关闭

def return_featureandmask(model, device,X, X_mask,store_path=None):
    
    confusion=ConfusionMatrix(num_classes=3,labels=[0,1,2])
    model.eval()

    all_out =[]
    all_y_pred = []
    all_y =[]
    
    X = X.to(device)
    X_mask = X_mask.to(device)
            
            # output = rnn_decoder(cnn_encoder(X))
     
    masks_maps ,feature_maps = model(X,X_mask)
    masks_maps_str = ['mask_4','mask_3','mask_1','mask_2']
    feature_maps_str = ['querys4','querys3','querys2']
    
    for num_layer in range(len(masks_maps)):
        plt.figure(figsize=(50, 10))
        for i in range(5):
            store_path_name =store_path+'/'+masks_maps_str[num_layer]+'_'+str(i)
            layer_viz = show_feature_map(masks_maps[num_layer][:,i,:,:,:],store_path_name)
        # layer_viz = layer_viz.data.cpu()
        
        
       
    return True

## -------------------- end of model prediction ---------------------- ##

###  0512  optical flow final #####

def optical_flow_early_fusion_final_prediction(model, device, loader,result_f):
    
    confusion=ConfusionMatrix(num_classes=3,labels=[0,1,2])
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()
    all_out =[]
    all_y_pred = []
    all_y =[]
    with torch.no_grad():
        for batch_idx, (X, y,X_mask,X_flow) in enumerate(tqdm(loader)):
            # distribute data to device
            # Z = Z.to(device)
            X = X.to(device)
            X_mask = X_mask.to(device)
            X_flow =X_flow.to(device)
            # output = rnn_decoder(cnn_encoder(X))
            output = cnn_encoder(X,X_flow)
            output = nn.Softmax(dim=1)(output)
            all_out+=output
            all_y+=y
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())
            confusion.update(np.array(y_pred.to("cpu")),y.to("cpu").numpy())
    acc_best_now=confusion.summary(result_f)
    # all_out = torch.stack(all_out,dim=0).to("cpu").view(-1,3)
    all_out = torch.stack(all_out,dim=0).to("cpu")
    all_y = torch.stack(all_y,dim=0).to("cpu")
    return all_y_pred,acc_best_now,all_out,all_y

def optical_flow_late_fusion_final_prediction(model, device, loader,result_f):
    
    confusion=ConfusionMatrix(num_classes=3,labels=[0,1,2])
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()
    all_out =[]
    all_y_pred = []
    all_y =[]
    with torch.no_grad():
        for batch_idx, (X, y,X_mask,X_flow) in enumerate(tqdm(loader)):
            # distribute data to device
            # Z = Z.to(device)
            X = X.to(device)
            X_mask = X_mask.to(device)
            X_flow =X_flow.to(device)
            # output = rnn_decoder(cnn_encoder(X))
            output_spatial =cnn_encoder(X,X_flow)   # output has dim = (batch, number of classes)
            output_temporal = rnn_decoder(X,X_flow)
            output = output_spatial + output_temporal
            # output = cnn_encoder(X,X_flow)
            # output = nn.Softmax(dim=1)(output)
            all_out+=output
            all_y+=y
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())
            confusion.update(np.array(y_pred.to("cpu")),y.to("cpu").numpy())
    acc_best_now=confusion.summary(result_f)
    # all_out = torch.stack(all_out,dim=0).to("cpu").view(-1,3)
    all_out = torch.stack(all_out,dim=0).to("cpu")
    all_y = torch.stack(all_y,dim=0).to("cpu")
    return all_y_pred,acc_best_now,all_out,all_y

def mask_late_fusion_final_prediction(model, device, loader,result_f):
    
    confusion=ConfusionMatrix(num_classes=3,labels=[0,1,2])
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()
    all_out =[]
    all_y_pred = []
    all_y =[]
    with torch.no_grad():
        for batch_idx, (X, y,X_mask,X_flow) in enumerate(tqdm(loader)):
            # distribute data to device
            # Z = Z.to(device)
            X = X.to(device)
            X_mask = X_mask.to(device)
            X_flow =X_flow.to(device)
            # output = rnn_decoder(cnn_encoder(X))
            output_spatial =cnn_encoder(X,X_mask)   # output has dim = (batch, number of classes)
            output_temporal = rnn_decoder(X,X_mask)
            output = output_spatial + output_temporal
            # output = cnn_encoder(X,X_flow)
            # output = nn.Softmax(dim=1)(output)
            all_out+=output
            all_y+=y
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())
            confusion.update(np.array(y_pred.to("cpu")),y.to("cpu").numpy())
    acc_best_now=confusion.summary(result_f)
    # all_out = torch.stack(all_out,dim=0).to("cpu").view(-1,3)
    all_out = torch.stack(all_out,dim=0).to("cpu")
    all_y = torch.stack(all_y,dim=0).to("cpu")
    return all_y_pred,acc_best_now,all_out,all_y
def mask_final_prediction(model, device, loader,result_f):
    
    confusion=ConfusionMatrix(num_classes=3,labels=[0,1,2])
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()
    all_out =[]
    all_y_pred = []
    all_y =[]
    with torch.no_grad():
        for batch_idx, (X, y,X_mask,X_flow) in enumerate(tqdm(loader)):
            # distribute data to device
            # Z = Z.to(device)
            X = X.to(device)
            X_mask = X_mask.to(device)
            X_flow =X_flow.to(device)
            # output = rnn_decoder(cnn_encoder(X))
            output = cnn_encoder(X,X_mask)
            output = nn.Softmax(dim=1)(output)
            all_out+=output
            all_y+=y
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())
            confusion.update(np.array(y_pred.to("cpu")),y.to("cpu").numpy())
    acc_best_now=confusion.summary(result_f)
    # all_out = torch.stack(all_out,dim=0).to("cpu").view(-1,3)
    all_out = torch.stack(all_out,dim=0).to("cpu")
    all_y = torch.stack(all_y,dim=0).to("cpu")
    return all_y_pred,acc_best_now,all_out,all_y


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
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
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
            x=x.view(x.size(0), -1)   #50 512     8*512


            #  0112 观察是否是attention层
            
            y=x.unsqueeze(1)     #第一个ECA
            y=self.eca1(y)
            y=y.squeeze(1)
            x=x*y   # 8*512

            # FC layers
            x = self.bn1(self.fc1(x)) #全连接
            x = F.relu(x) #8*512

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

        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)  #stack到29个，（1*512），变成29×512
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


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
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)
       
    def forward(self, x_RNN):
        # 为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous chunk(连续的块)
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



## -------------------focal loss ---------------------------------##
class FocalLoss(nn.Module):
    def __init__(self, gamma=0,alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target,device):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(0,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp()).to(device)

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()







if __name__ == '__main__':
    config = {}
    model = Generator(config).cuda()
    x = torch.randn(32, 3, 3, 128, 128).cuda()
    y, sim = model(x)
    print(y.size())
