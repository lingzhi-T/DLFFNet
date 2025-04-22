import os
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
# from networks.accuracy import roc
from scipy import interp
warnings.filterwarnings("ignore")
import pandas as pd
from torch.nn.functional import interpolate

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

        self.drop_out = nn.Dropout(0.5)
      

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

def save_checkpoint(state,filename="/home/tlz/GCCS_0916/checkpoint"):
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

def Conv3d_final_prediction(model, device, loader):
    model.eval()

    all_y_pred = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(tqdm(loader)):
            # distribute data to device
            X = X.to(device)
            output = model(X)
            y_pred = output.max(1, keepdim=True)[1]  # location of max log-probability as prediction
            all_y_pred.extend(y_pred.cpu().data.squeeze().numpy().tolist())

    return all_y_pred


def get_args():
    parser = argparse.ArgumentParser(description='Train the Net via shell',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fc1',type=int,default=512)   #ResNet的第一个fc
    parser.add_argument('--fc2', type=int, default=512) #第二个
    parser.add_argument('--emb',type=int,default=256)   #转为LSTM输入的尺寸
    parser.add_argument('--rnn_layer', type=int, default=1) #LSTM的数量
    parser.add_argument('--rnn_emb', type=int, default=512)  #LSTM是输出维度
    parser.add_argument('--rnn_fc', type=int, default=512)   #LSTM后面的FC维度
    parser.add_argument('--area',type=int,default=64)        #辅助area特征向量的长度
    parser.add_argument('--random',type=int,default=42)      #数据集分离的random int
    parser.add_argument('--test',type=float,default=0.3)     #测试集的比例
    return parser.parse_args()
args=get_args()



# latent dim extracted by 2D CNN  #512







def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    model_Generator, model_Discriminator = model
    model_Generator.train()
    model_Discriminator.train()

    losses = []
    train_loss = 0
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X,y,X_mask) in enumerate(train_loader):   # x是29*224*224，Z是面积变化曲线，480*640，y是标签
        # distribute data to device
        #print (batch_idx)
        # Z = Z.to(device)

        X, y ,X_mask= X.to(device), y.to(device).view(-1, ),X_mask.to(device)
        # X, y ,X_mask= X, y.view(-1, ),X_mask
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
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU
        # out = F.sigmoid(output)

        # toprediction = [0 if item <= 0.5 else 1 for item in out.squeeze()]
        # correct_num = sum([toprediction[idx] == y[idx] for idx in range(len(toprediction))])
        # print('training epoch:{}, correct_num={}, accu = {:.5f}'.format(epoch,correct_num,correct_num.item()/len(toprediction)))
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))
        
        loss.backward()
        optimizer.step()
    train_loss/= len(train_loader.dataset)
    return train_loss, scores


def validation(model, device, optimizer, test_loader,fold=0,
    acc_best=0,acc_best_epoch=0,roc_best=0,roc_best_epoch=0,result_f=None):
    global checkpoint_k_path
    global roc_k_path
    
    # set model as testing mode
    confusion=ConfusionMatrix(num_classes=3,labels=[0,1,2])
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()
   

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

            # distribute data to device
            # Z=Z.to(device)
            X, y ,X_mask= X.to(device), y.to(device).view(-1, ),X_mask.to(device)

            output = cnn_encoder(X,X_mask)


            creerion=nn.BCEWithLogitsLoss()
            loss = F.cross_entropy(output, y,reduction='sum') if k!=1 else creerion(output.squeeze(),y.float())
            test_loss += loss.item()                 # sum up batch loss
            output = nn.Softmax(dim=1)(output)
            y_pred = output.max(1, keepdim=True)[1]  #y_pred不是真正的输出，我懒得改了。真正的输出是 out，
            all_y.extend(y)
            all_y_pred.extend(y_pred)
            # out=F.sigmoid(output)   #真正的输出
            all_out+=output
            #print('out is ', out)
            # toprediction=[0 if item<=0.5 else 1 for item in out.squeeze()]
            y_sum=np.append(y_sum,y.to(torch.device('cpu')).numpy())
            # scores_sum=np.append(scores_sum,out.to(torch.device('cpu')).numpy())
            #print('correct_num =', float(correct_num.cpu().item())/len(toprediction))
            # correct_num = sum([toprediction[idx] == y[idx] for idx in range(len(toprediction))])
            # count+=correct_num
            # length+=len(toprediction)
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
    num_class =3 
    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(all_y[:, i], all_out[:, i])
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
 
    jpg_name = os.path.join(roc_k_path,  str(epoch)+ '.jpg')
    plt.savefig(jpg_name)
    plt.close()
    test_loss /= len(test_loader.dataset)

    # compute accuracy

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, test_score))

    # save Pytorch models of best record
    #save_model_path=save_model_path
    # if not os.path.exists(save_model_path):
    #     os.mkdir(save_model_path)
    if roc_best < roc_auc_now:
        roc_best =roc_auc_now
        roc_best_epoch=epoch
        torch.save(cnn_encoder.state_dict(), os.path.join(checkpoint_k_path, '{}_cnn_encoder_epoch{}.pth'.format(fold,epoch + 1)))  # save spatial_encoder
        torch.save(rnn_decoder.state_dict(), os.path.join(checkpoint_k_path, '{}_rnn_decoder_epoch{}.pth'.format(fold,epoch + 1)))  # save motion_encoder
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_k_path, '{}_optimizer_epoch{}.pth'.format(fold,epoch + 1)))      # save optimizer
    if acc_best < acc_best_now:
        acc_best =acc_best_now
        acc_best_epoch=epoch
        torch.save(cnn_encoder.state_dict(), os.path.join(checkpoint_k_path, '{}_cnn_encoder_epoch{}.pth'.format(fold,epoch + 1)))  # save spatial_encoder
        torch.save(rnn_decoder.state_dict(), os.path.join(checkpoint_k_path, '{}_rnn_decoder_epoch{}.pth'.format(fold,epoch + 1)))  # save motion_encoder
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_k_path, '{}_optimizer_epoch{}.pth'.format(fold,epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))
  
    return test_loss, test_score, acc_best ,acc_best_epoch,roc_best,roc_best_epoch

# training parameters
k = 2            # 类别
epochs = 29   # training epochs
batch_size = 1#50
res_size =112      # ResNet image size 224　　＃ＲｅｓＮｅｔ输入尺寸
dropout_p = 0.5

log_interval = 10   # interval for displaying training info

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU
print(device)
# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': True,'drop_last':False} if use_cuda else {}
all_names = []


train_transform = transforms.Compose([transforms.Resize([res_size, res_size]),                        
                                transforms.ToTensor(),
                                #  transforms.Normalize(mean=[59, 57, 62], std=[49, 48, 49]),
                                ])
transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                ])             


##  储存路径
root_path = 'sunsong\\0524-result'
cross_validation_roc ='cross-validation-roc'
cross_validation_checkpoint ='cross-validation-checkpoint'
result_path = os.path.join(root_path,'cross-validation-result')
roc_path = os.path.join(root_path,cross_validation_roc,'test')
checkpoint_path = os.path.join(root_path,cross_validation_checkpoint)

cross_txt_path= 'cross-validation-txt'
txt_path = os.path.join(root_path,cross_txt_path)
if not os.path.exists(roc_path):
    os.makedirs(roc_path)
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
if not os.path.exists(result_path):
    os.mkdir(result_path)



selected_frames = 5
learning_rate = 0.0005

mask_path = "aosax-3d-data-5frame-0510-predictmask" 
data_path  = 'aosax-3d-data-5frame-0510-crop'
fold= 2        ## 第几折
epochs = 30
seed =0
batch_size = 1
test_data_path = os.path.join(txt_path,'0510-3d-4-classification-test_patient'+'.txt')
test_pathes = []
test_labeles = []

break_out=0
for file in os.listdir(data_path):
    break_out +=1
    patient = file
    with open(test_data_path) as read:
                        test_img_list = [line.strip() 
                                         for line in read.readlines()]
    assert len(test_img_list) > 0, "in '{}' file does not find any information.".format(test_img_list)    #aosax 2321
    if patient in set(test_img_list):
            path = os.path.join(data_path,file)
            label = file.split('_')[0]
            if label == '3':
                label ='1'
            elif label == '2' :
                label == '1'
            test_pathes.append(file)
            test_labeles.append(float(label))

test_set =  Dataset_CRNN_mask(data_path, mask_path,test_pathes, test_labeles, selected_frames, transform=(transform,train_transform))

data_loader = data.DataLoader(test_set ,**params)

mask_path = './sunsong/0000.png'
transform = transforms.Compose([
                                transforms.ToTensor(),
                                ])

image = Image.open(mask_path)
image = interpolate(transform(image).unsqueeze(0),size=[8,8]).squeeze()
image =image.numpy()>0
mask_image = np.ones(image.shape)
mask_result = mask_image*image


checkpoint_path = './sunsong/check_predictions'
save_model_name ='0530('+str(seed)+')_fpn-resnet18-4层local_feature_fusion-changemask_'+'3 classification'+str(batch_size)+'_'+str(learning_rate)+'_'+'checkpoint'+'_'+str(fold)+'_'+'folds'

save_model_path =os.path.join(checkpoint_path,save_model_name)
roc_path =os.path.join(roc_path , save_model_name)
if not os.path.exists(roc_path):
    os.mkdir(roc_path)
#
for epoch in range(epochs,epochs+1):
    
# for epoch in range(epochs):
    
    # test_data_path = os.path.join(txt_path,'0510-3d-4-classification-val_patient_'+str(fold)+'.txt')
    
    # 构建模型
    #  替换点resnet18_fpn_branch_wrefindices_changemask_attention为初始模型，但转换失败，resnet18_two_branch为缩减后的模型
    # model_Generator=  resnet18_fpn_branch_wrefindices_changemask_attention(num_classes=3,mask =mask_result,device =device).to(device)
    model_Generator = resnet18_two_branch(num_classes=3,device =device).to(device)

    #  resnet18_fpn_branch_wrefindices_changemask_attention的pth
    # pretain_model_path = 'c:/Users/Administrator/tlz/GCCS/sunsong/check_predictions/0530(0)_fpn-resnet18-4层local_feature_fusion-changemask_3 classification8_0.0005_checkpoint_1_folds/1_cnn_encoder_epoch21.pth'
    # resnet18_two_branch的pth
    pretain_model_path ='sunsong\\0524-result\\cross-validation-checkpoint\\0130_resnet18_two_branch3-classification8_0.005_checkpoint_2_folds\\2_cnn_encoder_epoch8.pth'
    model_dict=model_Generator.state_dict()
            # 1. filter out unnecessary keys
            # print(model_dict.keys())
    pretrained_dict = torch.load(pretain_model_path)    
    # pretrained_dict = {k: v for k, v in pretrained_dict.items()  if k.find('conv4')== -1  }
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_Generator.load_state_dict(model_dict)
    # model_Discriminator =Discriminator(config['dis']).to(device)
    # model_Generator.load_state_dict(torch.load(os.path.join(save_model_path, str(fold)+'_cnn_encoder_epoch'+str(epoch)+'.pth')))
    

    print('model reloaded!')

    print("Using", torch.cuda.device_count(), "GPU!")

    print('Predicting all {} videos:'.format(len(data_loader.dataset)))
    #替换点 
    result_f =open(os.path.join(result_path,'0627_crop-changemask分支-img_4层-权重-local-feature-fusion-resindice-mask+lstm+'+'3 classification'+str(selected_frames)+'_'+str(learning_rate)+'_'+'result'+'_'+'.txt'),"a+")
    
    # result_f =open(os.path.join(result_path,'0627_1层-权重-local-feature-fusion-mask分支'+'3 classification'+str(selected_frames)+'_'+str(learning_rate)+'_'+'result'+'_'+'.txt'),"a+")
    # result_f =open(os.path.join(result_path,'0627_4层-权重-local-feature-fusion-mask分支'+'3 classification'+str(selected_frames)+'_'+str(learning_rate)+'_'+'result'+'_'+'.txt'),"a+")
    
    all_y_pred,acc_best_now,all_out,all_y = final_prediction(model_Generator, device, data_loader,result_f=result_f)


    # write in pandas dataframe
    # df = pd.DataFrame(data={'y':all_y , 'y_pred':  all_y_pred})
    micro_auc,macro_auc = roc(all_out, all_y,roc_path,epochs)
                
    print(f'\nepoch\t{epoch:.4f} micro_auc\t{micro_auc:.4f}\tmacro_auc\t{macro_auc:.4f}')
    result_f.write(f'\nepoch\t{epoch} micro_auc\t{micro_auc:.4f}\tmacro_auc\t{macro_auc:.4f}')
    print('video prediction finished!')