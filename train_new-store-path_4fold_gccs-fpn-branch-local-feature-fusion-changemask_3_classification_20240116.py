import os
from cv2 import determinant
import numpy as np
from pandas.core import frame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
#import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import argparse
from tqdm import tqdm
import sys
from sklearn.model_selection import KFold
import warnings
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
# from networks.lofgan import *
from our_model import resnet18_fpn_branch_wrefindices_changemask_attention
import random
import torch.nn.functional as F
import torch
import os
# 初始化模型类
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch.nn.functional import interpolate
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn


def roc(results, gt_labels,roc_path,epoch):
    # global epoch 
    epoch=epoch
    # all_out  = [nn.Softmax(dim=-1)(torch.tensor(result)) for result in results]
    # all_out = results.view(-1,3)
    all_out =results
    all_y = [torch.tensor(y) for y in gt_labels]  
    all_y = torch.stack(all_y, dim=0)
    all_y= label_binarize(all_y, classes=[0, 1, 2])
    num_class =3  
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(all_y[:, i], all_out[ :,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(all_y.ravel(), all_out.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
      ### compute macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    roc_auc_now = roc_auc["micro"] 
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','navy'])
     ######  3分类修改
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.show()
 
    jpg_name = os.path.join(roc_path,  str(epoch)+ '.jpg')
    if not os.path.exists(roc_path):
        os.mkdir(roc_path)
    micro_fpr_path = os.path.join(roc_path,  str(epoch)+ '_microfpr.npy')
    micro_tpr_path = os.path.join(roc_path,str(epoch)+'_microtpr.npy')
    np.save(micro_fpr_path,fpr['micro'])
    np.save(micro_tpr_path,tpr['micro'])
    macro_fpr_path = os.path.join(roc_path,  str(epoch)+ '_macrofpr.npy')
    macro_tpr_path = os.path.join(roc_path,str(epoch)+'_macrotpr.npy')
    np.save(macro_fpr_path,fpr['macro'])
    np.save(macro_tpr_path,tpr['macro'])
    plt.savefig(jpg_name)

    return  roc_auc["micro"],roc_auc["macro"]
    pass

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            # elif pool_type == 'lse':
            #     # LSE pool only
            #     lse_pool = logsumexp_2d(x)
            #     channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False, use_cbam=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        # elif norm == 'adain':
        #     self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if norm == 'sn':
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ks, st,padding=padding,bias=self.use_bias)

        if use_cbam:
            self.cbam = CBAM(out_dim, 16, no_spatial=True)
        else:
            self.cbam = None

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            # x = self.conv(self.pad(x))
            x = self.conv(x)
            
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(x)
            if self.norm:
                x = self.norm(x)
            if self.cbam:
                x = self.cbam(x)
            if self.activation:
                x = self.activation(x)
        return x
    
class resnet18_two_branch(nn.Module):
    def __init__(self, num_classes,device):
        super(resnet18_two_branch, self).__init__()
        self.device =device
        resnet = models.resnet18(pretrained=True)
        self.mask_head = nn.Sequential(Conv2dBlock(1, 1, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero'),
                             Conv2dBlock(1, 1, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
                             )
        self.resnet_head=nn.Sequential(*list(resnet.children())[:5])
        self.conv2d_1 = nn.Sequential(*list(resnet.children())[5])
        self.mask_conv2d_1 = Conv2dBlock(1, 1, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')        
        self.conv2d_2 = nn.Sequential(*list(resnet.children())[6])
        self.mask_conv2d_2 = Conv2dBlock(1, 1, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        self.conv2d_3 = nn.Sequential(*list(resnet.children())[7])
        self.mask_conv2d_3 = Conv2dBlock(1, 1, 3, 2, 1,
                             norm='bn',
                             activation='lrelu',
                             pad_type='zero')
        n_class = num_classes
        self.pool = nn.AvgPool3d(kernel_size=(1,4,4))             
        self.linear = nn.Linear(4*n_class,n_class)
        self.LSTM  = nn.LSTM(
            input_size=512,
            hidden_size=256,        
            num_layers=1,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.fc1 = nn.Linear(256, 8) ##changemask
        self.fc2 = nn.Linear(13, n_class)  ## changemask分支时替换    
        self.fc_loc=nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(True),
            nn.Linear(16,6)
        )
        self.linear512 = nn.Linear(512*5,256)

        self.drop_out = nn.Dropout(0.1)
      

    def forward(self, xs,mask):
        ### 0523 集成化
        b, k, C, H, W = xs.size()
        device = self.device
        ## 初始相似性  
        similarity_total = torch.cat([torch.ones(b,1) for _ in range(k)],dim=1).to(device)
        similarity_sum = torch.sum(similarity_total, dim=1, keepdim=True).expand(b, k)
        self.similarity = similarity_total / similarity_sum  # b*k
        b, k, m_C, H,W = mask.size()     
        mask_orginal = mask.view(-1,m_C,H,W)
        xs = xs.view(-1, C, H, W)
        querys = self.resnet_head(xs)         
        c, h, w = querys.size()[-3:]
        querys = querys.view(b, k, c, h, w)      

        mask = interpolate(mask_orginal,size=[h,w])  # 第一阶段插值
        mask = mask.view(b, k, m_C, h, w)      
        
        querys1=querys.clone()

        ###  第二阶段
        querys1 = self.conv2d_1(querys1.view(-1,c,h,w))
        mask = self.mask_conv2d_1(mask.view(-1,m_C,h,w))       
        c, h, w = querys1.size()[-3:]
        querys1 = querys1.view(b, k, c, h, w)
        mask = mask.view(b, k,m_C , h, w)

        querys2=querys1.clone()
        querys2 = self.conv2d_2(querys2.view(-1,c,h,w))
        querys3 =querys2.clone()
        mask = self.mask_conv2d_2(mask.view(-1,m_C,h,w))

        c, h, w = querys2.size()[-3:]
        querys2 = querys2.view(b, k, c, h, w)
        mask = mask.view(b, k,m_C , h, w)
        querys3 = self.conv2d_3(querys3.view(-1,c,h,w))
        mask = self.mask_conv2d_3(mask.view(-1,m_C,h,w))
        ## 0402 转换成函数
        c, h, w = querys3.size()[-3:]
        querys3 = querys3.view(b, k, c, h, w)
        mask = mask.view(b, k,m_C , h, w)

        
        querys4 = self.pool(querys3).squeeze()#.unsqueeze(0)

        mask = self.pool(mask).squeeze().view(b,5)
        # querys4 = querys4.squeeze(dim=0)
        ## 0402 转换成函数
        # RNN_out, (h_n, h_c) = self.LSTM(querys4, None)
        # # RNN_out = querys4
        # x= RNN_out[:,-1,]
        querys4 = querys4.view(b,k*512)
        x = self.linear512(querys4)

        x = self.fc1(x)
        x = F.relu(x)
        # x=self.drop_out(x)
        x= torch.cat((x,mask),dim=1)
        x = self.fc2(x)
        return x


from scipy import interp
warnings.filterwarnings("ignore")
## 1.首先跑起来，后面简化版本

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

def save_checkpoint(state,filename="/home/tlz/GCCS/checkpoint"):
    print("=> Saving checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

#  0925 查找roc曲线中最好的阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    model_Generator = model
    model_Generator.train()
    losses = []
    train_loss = 0
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X,y,X_mask) in enumerate(train_loader):   # x是29*224*224，Z是面积变化曲线，480*640，y是标签
        X, y ,X_mask= X.to(device), y.to(device).view(-1, ),X_mask.to(device)
        N_count += X.size(0)
        optimizer.zero_grad()
        output =model_Generator(X,X_mask)   # output has dim = (batch, number of classes)
        #output =F.sigmoid(output)
        creerion=nn.BCEWithLogitsLoss()  #二分类....字幕拼错了懒得改了..
        loss = F.cross_entropy(output, y) if k!=1 else creerion(output.squeeze(),y.float())
        train_loss += loss.item()
        losses.append(loss.item())
        # to compute accuracy
        y_pred = torch.max(output,1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.numpy(), y_pred.cpu().data.numpy())
        scores.append(step_score)         # computed on CPU
        # out = F.sigmoid(output)
        # toprediction = [0 if item <= 0.5 else 1 for item in out.squeeze()]
        # correct_num = sum([toprediction[idx] == y[idx] for idx in range(len(toprediction))])
        # print('training epoch:{}, correct_num={}, accu = {:.5f}'.format(epoch,correct_num,correct_num.item()/len(toprediction)))
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                    epoch , N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))        
        loss.backward()
        optimizer.step()
    train_loss/= len(train_loader.dataset)
    print('Train loss :{}'.format(train_loss))
    return train_loss, scores


def validation(model, device, optimizer, test_loader,fold=0,
    acc_best=0,acc_best_epoch=0,roc_best=0,roc_best_epoch=0,result_f=None):
    global checkpoint_k_path
    global roc_k_path
    # set model as testing mode
    confusion=ConfusionMatrix(num_classes=3,labels=[0,1,2])
    cnn_encoder= model
    cnn_encoder.eval()
    test_loss = 0
    all_y = []
    all_y_pred = []
    y_sum=[]
    y_sum=np.array(y_sum)
    scores_sum=[]
    scores_sum=np.array(scores_sum)
    count=0
    length=0
    all_out=[]
    with torch.no_grad():
        for X, y,X_mask in test_loader:
            X, y ,X_mask= X.to(device), y.to(device).view(-1,),X_mask.to(device)
            output = cnn_encoder(X,X_mask)
            creerion=nn.BCEWithLogitsLoss()
            loss = F.cross_entropy(output, y,reduction='sum') if k!=1 else creerion(output.squeeze(),y.float())
            test_loss += loss.item()                 # sum up batch loss
            output = nn.Softmax(dim=1)(output)
            y_pred = output.max(1, keepdim=True)[1]  #y_pred不是真正的输出，我懒得改了。真正的输出是 out，
            all_y.extend(y)
            all_y_pred.extend(y_pred)         
            all_out+=output      
            y_sum=np.append(y_sum,y.to(torch.device('cpu')).numpy())     
            confusion.update(np.array(y_pred.to("cpu")),y.to("cpu").numpy())
    all_out = torch.stack(all_out,dim=0).to("cpu")
    all_y = torch.stack(all_y, dim=0).to("cpu")
    all_y_pred = torch.stack(all_y_pred, dim=0).to("cpu")
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    acc_best_now=confusion.summary(result_f)
    result_f.write('\n')
    sentence='epoch = '+str(epoch)
    result_f.write(sentence)
    result_f.write('\n')
    ###  ------------------------------------roc  -----------------------------------------------------###
    ## 标签二值化
    all_y= label_binarize(all_y.to("cpu"), classes=[0, 1, 2])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    ######  3分类修改
    num_class = 3  
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(all_y[:, i], all_out[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    ## 这是三分类的
    fpr["micro"], tpr["micro"], _ = roc_curve(all_y.ravel(), all_out.ravel())
    ## 这是二分类的
    # fpr["micro"], tpr["micro"], _ = roc_curve(all_y.ravel(), all_out[:,0].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    ### compute macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    roc_auc_now = roc_auc["micro"] 
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','navy'])
     ######  3分类修改
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.show()
 
    jpg_name = os.path.join(roc_k_path,  str(epoch)+ '.jpg')
    micro_fpr_path = os.path.join(roc_k_path,  str(epoch)+ '_microfpr.npy')
    micro_tpr_path = os.path.join(roc_k_path,str(epoch)+'_microtpr.npy')
    np.save(micro_fpr_path,fpr['micro'])
    np.save(micro_tpr_path,tpr['micro'])
    macro_fpr_path = os.path.join(roc_k_path,  str(epoch)+ '_macrofpr.npy')
    macro_tpr_path = os.path.join(roc_k_path,str(epoch)+'_macrotpr.npy')
    np.save(macro_fpr_path,fpr['macro'])
    np.save(macro_tpr_path,tpr['macro'])
    plt.savefig(jpg_name)
    plt.savefig(jpg_name)
    plt.close()
    test_loss /= len(test_loader.dataset)

    # compute accuracy

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, test_score))


    if roc_best <= roc_auc_now:
        roc_best =roc_auc_now
        roc_best_epoch=epoch
        torch.save(cnn_encoder.state_dict(), os.path.join(checkpoint_k_path, '{}_cnn_encoder_epoch{}.pth'.format(fold,epoch )))  # save spatial_encoder
        # torch.save(rnn_decoder.state_dict(), os.path.join(checkpoint_k_path, '{}_rnn_decoder_epoch{}.pth'.format(fold,epoch )))  # save motion_encoder
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_k_path, '{}_optimizer_epoch{}.pth'.format(fold,epoch )))      # save optimizer
    if acc_best <= acc_best_now:
        acc_best =acc_best_now
        acc_best_epoch=epoch
        torch.save(cnn_encoder.state_dict(), os.path.join(checkpoint_k_path, '{}_cnn_encoder_epoch{}.pth'.format(fold,epoch + 1)))  # save spatial_encoder
        # torch.save(rnn_decoder.state_dict(), os.path.join(checkpoint_k_path, '{}_rnn_decoder_epoch{}.pth'.format(fold,epoch + 1)))  # save motion_encoder
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_k_path, '{}_optimizer_epoch{}.pth'.format(fold,epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch ))
  
    return test_loss, test_score, acc_best ,acc_best_epoch,roc_best,roc_best_epoch

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


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
os.environ['CUDA_VISIBLE_DEVICE']='0,1,2,3,4,5,6,7'
device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU
print(device)
# Data loading parameters
# convert labels -> category
le = LabelEncoder()
            

## mask 0324

mask_path = './sunsong/0000.png'
transform = transforms.Compose([
                                transforms.ToTensor(),
                                ])
mask_result = transform(Image.open(mask_path)).unsqueeze(0)

# training parameters
data_path = "./sunsong\\data\\aosax-3d-data-5frame-0510-crop"   # 0409 没有任何处理的原图路径 
mask_path = "./sunsong\\data\\aosax-3d-data-5frame-0510-predictmask"  
res_size =112      # ResNet image size 224　　＃ＲｅｓＮｅｔ输入尺寸
dropout_p = 0.5
k = 3            # 类别
epochs = 100   # training epochs
batch_size = 16 #50
learning_rate = 0.005
log_interval = 5   # interval for displaying training inf
seed = 0
zheshu = 4  ##  删减为只有第一折
params = {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True,'drop_last':False} if use_cuda else {}
train_transform = transforms.Compose([transforms.Resize([res_size, res_size]),                            
                                transforms.ToTensor(),
                                # transforms.Normalize(mean=[59, 57, 62], std=[49, 48, 49]),
                                ])
transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                ])                   
#随机数设置
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.deterministic = True
setup_seed(seed)
##  储存路径
root_path = './sunsong/0524-result'
cross_validation_roc ='cross-validation-roc'
cross_validation_checkpoint ='cross-validation-checkpoint'
result_path = os.path.join(root_path,'cross-validation-result')
roc_path = os.path.join(root_path,cross_validation_roc)
checkpoint_path = os.path.join(root_path,cross_validation_checkpoint)
cross_txt_path= 'cross-validation-txt'
txt_path = os.path.join(root_path,cross_txt_path)
if not os.path.exists(roc_path):
    os.mkdir(roc_path)
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
if not os.path.exists(result_path):
    os.mkdir(result_path)

kfolds =5 
for fold in range(1,kfolds):
    if fold ==zheshu :
        acc_best =0
        acc_best_epoch =0
        roc_best =0
        roc_best_epoch =0
        selected_frames = 5
        # result_f =open(os.path.join(result_path,'0530('+str(seed)+')_fpn-resnet18-4层local_feature_fusion-插值changemask_'+'3 classification'+str(batch_size)+'_'+str(learning_rate)+'_'+'result'+'_'+str(fold)+'.txt'),"a+")
        # print(f'第{fold}折交叉验证')
        # result_f.write(f"第{fold}折交叉验证")
        # roc_k_path = os.path.join(roc_path,'0530('+str(seed)+')_fpn-resnet18-4层local_feature_fusion-插值changemask_'+'3 classification'+str(batch_size)+'_'+str(learning_rate)+'_'+'roc'+'_'+str(fold)+'_'+'folds')
        # if not os.path.exists(roc_k_path):
        #     os.mkdir(roc_k_path)
        # checkpoint_k_path = os.path.join(checkpoint_path,'0530('+str(seed)+')_fpn-resnet18-4层local_feature_fusion-插值changemask_'+'3 classification'+str(batch_size)+'_'+str(learning_rate)+'_'+'checkpoint'+'_'+str(fold)+'_'+'folds')
        ##0702超参数敏感实验
        result_f =open(os.path.join(result_path,'0130_resnet18_two_branch'+'3-classification'+str(batch_size)+'_'+str(learning_rate)+'_'+'result'+'_'+str(fold)+'.txt'),"a+")
        print(f'第{fold}折交叉验证')
        result_f.write(f"第{fold}折交叉验证")
        roc_k_path = os.path.join(roc_path,'0130_resnet18_two_branch'+'3-classification'+str(batch_size)+'_'+str(learning_rate)+'_'+'roc'+'_'+str(fold)+'_'+'folds')
        if not os.path.exists(roc_k_path):
            os.mkdir(roc_k_path)
        checkpoint_k_path = os.path.join(checkpoint_path,'0130_resnet18_two_branch'+'3-classification'+str(batch_size)+'_'+str(learning_rate)+'_'+'checkpoint'+'_'+str(fold)+'_'+'folds')
        if not os.path.exists(checkpoint_k_path):
            os.mkdir(checkpoint_k_path)     
        train_data_path = os.path.join(txt_path,'0510-3d-4-classification-train_patient_'+str(fold)+'.txt')
        test_data_path = os.path.join(txt_path,'0510-3d-4-classification-val_patient_'+str(fold)+'.txt')
        train_pathes = []
        train_labeles = []
        test_pathes = []
        test_labeles = []
        break_out=0
        for file in os.listdir(data_path):
            break_out +=1
            patient = file
           #  在train 和 test 的txt 里面进行配对
            with open(train_data_path) as read:
                    train_img_list = [line.strip() 
                                     for line in read.readlines()]
            assert len(train_img_list) > 0, "in '{}' file does not find any information.".format(train_img_list)    #aosax 8549

            if patient in set(train_img_list):
                    path = os.path.join(data_path,file)
                    label = file.split('_')[0]
                    if label == '3':
                        label ='2'
                    elif label == '2' :
                        label = '1'
                    train_pathes.append(file)
                    train_labeles.append(float(label))

            with open(test_data_path) as read:
                    test_img_list = [line.strip() 
                                     for line in read.readlines()]
            assert len(test_img_list) > 0, "in '{}' file does not find any information.".format(train_img_list)    #aosax 2321
            if patient in set(test_img_list):
                    path = os.path.join(data_path,file)
                    label = file.split('_')[0]
                    if label == '3':
                        label = '2'
                    elif label == '2' :
                        label = '1'
                    test_pathes.append(file)
                    test_labeles.append(float(label))
        train_set, valid_set = Dataset_CRNN_mask(data_path,mask_path, train_pathes, train_labeles, selected_frames, transform=(transform,train_transform)), \
                               Dataset_CRNN_mask(data_path, mask_path,test_pathes, test_labeles, selected_frames, transform=(transform,train_transform))               
        train_loader = data.DataLoader(train_set ,**params)
        valid_loader = data.DataLoader(valid_set, **params)
        
        model_Generator = resnet18_fpn_branch_wrefindices_changemask_attention(num_classes=3,mask =mask_result,device =device).to(device)
        # model_Generator = resnet18_two_branch(num_classes=3,device =device).to(device)
        print("Using", torch.cuda.device_count(), "GPU!")

        epoch_train_losses = []
        epoch_train_scores = []
        epoch_test_losses = []
        epoch_test_scores = []
        model_params = list(model_Generator.parameters())
        optimizer = torch.optim.Adam(model_params, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.8, patience=3, verbose=True
        )
        # start training
        for epoch in tqdm(range(epochs)):           
            train_losses, train_scores  = train(log_interval, model_Generator, device, train_loader, optimizer, epoch)
            epoch_test_loss, epoch_test_score ,acc_best ,acc_best_epoch,roc_best,roc_best_epoch= validation(model_Generator, device, optimizer, 
            valid_loader,fold,acc_best=acc_best,
            acc_best_epoch=acc_best_epoch,roc_best=roc_best,roc_best_epoch=roc_best_epoch,result_f=result_f)
            result_f.write(f"\nepoch{epoch}kfold:{k},acc_best:{acc_best},acc_best_epoch:{acc_best_epoch},roc_best:{roc_best},roc_best_epoch:{roc_best_epoch}")
            scheduler.step(train_losses)
            #因为数据集小，以测试精度未标注，手动降低学习率
            epoch_train_losses.append(train_losses)
            epoch_train_scores.append(train_scores)
            epoch_test_losses.append(epoch_test_loss)
            epoch_test_scores.append(epoch_test_score)
            # save all train test results
            A = np.array(epoch_train_losses)
            B = np.array(epoch_train_scores)
            C = np.array(epoch_test_losses)
            D = np.array(epoch_test_scores)
          
    fold+=1
   


