import torch
import torchvision.models as models
import torchvision.transforms as transforms

import torch.nn.functional as F
from torch import nn

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import interpolate
import numpy as np
from PIL import Image
import random
class MultiConv2DBlock(nn.Module):
    def __init__(self, fin, fout, fhid=None,
                 activation='lrelu', norm='none'):
        super(MultiConv2DBlock, self).__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = max(fin, fout) if fhid is None else fhid
        self.conv_0 = Conv2dBlock(self.fin, self.fhid, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=False)
        self.conv_1 = Conv2dBlock(self.fhid, self.fout, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=False)
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(self.fin, self.fout, 1, 1,
                                      activation='none', use_bias=False)

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out


class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim,
                                    norm=norm,
                                    activation=activation,
                                    pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', use_cbam=False):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        self.model = nn.Sequential(*model)
        if use_cbam:
            self.cbam = CBAM(dim, 16, no_spatial=True)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x
        out = self.model(x)
        if self.cbam:
            out = self.cbam(x)
        out += residual
        return out


class ActFirstResBlock(nn.Module):
    def __init__(self, fin, fout, fhid=None,
                 activation='lrelu', norm='none'):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = max(fin, fout) if fhid is None else fhid
        self.conv_0 = Conv2dBlock(self.fin, self.fhid, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        self.conv_1 = Conv2dBlock(self.fhid, self.fout, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(self.fin, self.fout, 1, 1,
                                      activation='none', use_bias=False)

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
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

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


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
            self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

        if use_cbam:
            self.cbam = CBAM(out_dim, 16, no_spatial=True)
        else:
            self.cbam = None

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.cbam:
                x = self.cbam(x)
            if self.activation:
                x = self.activation(x)
        return x


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
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


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

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views)
    index = index.expand(expanse)
    return torch.gather(input, dim, index)




def batched_scatter(input, dim, index, src):
    views = [input.shape[0]] + [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views)
    index = index.expand(expanse)
    return torch.scatter(input, dim, index, src)

def find_max_length(mask,b,n,h,w):
    max_length = []
    for index in range(n):
        base_mask =mask[:, index, :, :, :]
        image_basemask =base_mask.squeeze()
        image =image_basemask.cpu().detach().numpy()>0
        mask_image = np.ones(image.shape)
        mask_result_basemask = mask_image*image
        feat_immediate_basemask=torch.tensor(mask_result_basemask).reshape(b,h*w)  
        for i in range(b):
            max_length.append(torch.where(feat_immediate_basemask[i,:])[0].size(0))
    return max(max_length)


def batched_mask_stack(base_mask,b,h,w,max_batch_length):
    image_basemask = interpolate(base_mask,size=[h,w]).squeeze()
    image =image_basemask.cpu().numpy()>0
    mask_image = np.ones(image.shape)
    mask_result_basemask = mask_image*image
    feat_immediate_basemask=torch.tensor(mask_result_basemask).reshape(b,h*w)  
    # indices=torch.where(feat_immediate_basemask[i:,]) for i in range(b)]
    immediate =[]
    max_length = []
    for i in range(b):
       immediate.append(torch.where(feat_immediate_basemask[i,:]))
       max_length.append(immediate[i][0].size(0))
       if max_length[i] == 0:
           mask_path = '/media/tlz/5beb59b1-c3c0-481a-8ad8-0452b4243a70/home/tlz/labelme-master/examples/semantic_segmentation/data_dataset_voc/SegmentationClassPNG/0000.png'
           transform = transforms.Compose([
                                transforms.ToTensor(),
                                ])
           mask_result = transform(Image.open(mask_path)).unsqueeze(0)
           image = interpolate(mask_result,size=[h,w]).squeeze()
           image =image.cuda().numpy()>0
           mask_image = np.ones(image.shape)
           mask_result = mask_image*image
            # feat_immediate=torch.tensor(mask_result).resize(b*h*w)         
           feat_immediate=torch.tensor(mask_result).resize(h*w)   
           immediate[i]=( torch.where(feat_immediate))
           max_length[i]=(torch.where(feat_immediate)[0].size(0))
    #    print(len(immediate[i]))
    # indices=torch.cat(torch.stack(immediate,dim=0)).cuda()
    # max_batch_length = max(max_length)
    indices =[]
    for i in range(b):
        if max_length[i]< max_batch_length:
            support = max_batch_length- max_length[i]
            if max_length[i]==0:
                
                indices.append(torch.LongTensor(random.sample(range(h*w), max_batch_length)))
            else:    
                remainder =support %max_length[i]
                multi = int(support/max_length[i])
                media_result =  immediate[i][0]
                for k  in range(multi):
                    media_result=torch.cat((media_result,immediate[i][0]))
                if multi == 0:
                    # print(immediate[i][0][:2])
                    # print(immediate[i][0])
                    turple = (immediate[i][0],immediate[i][0][:remainder])
                    indices.append(torch.cat(turple))
                else:
                    indices.append(torch.cat((media_result,immediate[i][0][:remainder])))
        else:
            indices.append(immediate[i][0])
    indices= torch.stack(indices,dim=0)
    return indices


def batched_base_mask_stack(base_mask,b,h,w,max_batch_length):
    ### mask 堆叠
    image_basemask = interpolate(base_mask,size=[h,w]).squeeze()
    image = image_basemask>0
    # image =image_basemask.cpu().detach().numpy()>0
    # mask_image = np.ones(image.shape)
    device= image.device
    mask_image = torch.ones(image.shape).to(image.device)
    mask_result_basemask = mask_image*image
    feat_immediate_basemask=torch.tensor(mask_result_basemask).reshape(b,h*w)  
    # indices=torch.where(feat_immediate_basemask[i:,]) for i in range(b)]
    immediate =[]
    max_length = []  ## 寻找一个batch里的最大长度
    for i in range(b):
        ## 0627消融实验  index mean 05.0.25.0.1 超参数
    #    print(feat_immediate_basemask[i,:].mean())
       immediate.append(torch.where(feat_immediate_basemask[i,:]>(feat_immediate_basemask[i,:].mean())))
       feat_immediate_basemask_size= feat_immediate_basemask.shape[1]
       mask_percentile = 0.1

    #    immediate.append((feat_immediate_basemask[i,:].topk(int(mask_percentile*feat_immediate_basemask_size))[1],))
    #
       max_length.append(immediate[i][0].size(0))
       
       ##  异常处理  用中心大mask替代
       if max_length[i] == 0:
           mask_path = 'sunsong\\0001.png'
           transform = transforms.Compose([
                                transforms.ToTensor(),
                                ])
           mask_result = transform(Image.open(mask_path)).unsqueeze(0)
           image = interpolate(mask_result,size=[h,w]).squeeze()
           mask_image = torch.ones(image.shape)
        #    image =image.cpu().numpy()>0
        #    mask_image = np.ones(image.shape)
           mask_result = mask_image*image
            # feat_immediate=torch.tensor(mask_result).resize(b*h*w)         
           feat_immediate=torch.tensor(mask_result).resize(h*w).to(device)   
           immediate[i]=( torch.where(feat_immediate>feat_immediate.mean()))
           max_length[i]=(torch.where(feat_immediate)[0].size(0))
    #    print(len(immediate[i]))
    # indices=torch.cat(torch.stack(immediate,dim=0)).cuda()
    # max_batch_length = max(max_length)
    indices =[]
    for i in range(b):
        if max_length[i]< max_batch_length:
            support = max_batch_length- max_length[i]
            if max_length[i]==0:
                ### 随机索引
                indices.append(torch.LongTensor(random.sample(range(h*w), max_batch_length)).to(device))
            else:    
                remainder =support %max_length[i]
                multi = int(support/max_length[i])
                media_result =  immediate[i][0]
                for k  in range(multi):
                    media_result=torch.cat((media_result.to(device),immediate[i][0].to(device)))
                if multi == 0:
                    # print(immediate[i][0][:2])
                    # print(immediate[i][0])
                    turple = (immediate[i][0],immediate[i][0][:remainder])
                    indices.append(torch.cat(turple).to(device))
                else:
                    indices.append(torch.cat((media_result,immediate[i][0][:remainder])).to(device))
        else:
            indices.append(immediate[i][0].to(device))
    indices= torch.stack(indices,dim=0)
    return indices

class LocalFusionModule_mask(nn.Module):
    def __init__(self, inplanes, rate,mask,CNN_embed_dim=128,h_RNN_layers=3, h_RNN=128):
        super(LocalFusionModule_mask, self).__init__()

        
        self.W = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inplanes)
        )
        self.rate = rate
        self.mask = mask
        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN
        self.LSTM  = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
    def forward(self, feat, refs, index, similarity,mask):
        
        refs = torch.cat([refs[:, :index, :, :, :], refs[:, (index + 1):, :, :, :]], dim=1)
        base_similarity = similarity[:, index]
        ref_similarities = torch.cat([similarity[:, :index], similarity[:, (index + 1):]], dim=1)

        # take ref:(32, 2, 128, 8, 8) for example
        b, n, c, h, w = refs.size()
        refs = refs.view(b * n, c, h, w)

        w_feat = feat.view(b, c, -1)
        w_feat = w_feat.permute(0, 2, 1).contiguous()
        w_feat = F.normalize(w_feat, dim=2)  # (32*64*128)

        w_refs = refs.view(b, n, c, -1)
        w_refs = w_refs.permute(0, 2, 1, 3).contiguous().view(b, c, -1)
        w_refs = F.normalize(w_refs, dim=1)  # (32*128*128)
        max_index_length = find_max_length(mask,b,n+1,h,w)
        # local selection
        # rate = self.rate
        # num = int(rate * h * w)
        
        # feat_indices = torch.cat([torch.LongTensor(random.sample(range(h * w), num)).unsqueeze(0) for _ in range(b)],
        #                          dim=0).cuda()  # B*num
         #  0324 mask设置
        base_mask = mask[:, index, :, :, :]
        indices = batched_mask_stack(base_mask,b,h,w,max_index_length).cuda()
        
          #  0324 mask设置-*------------------------------------------
        # image = interpolate(self.mask,size=[h,w]).squeeze()
        # image =image.cpu().numpy()>0
        # mask_image = np.ones(image.shape)
        # mask_result = mask_image*image
        # # feat_immediate=torch.tensor(mask_result).resize(b*h*w)         
        # feat_immediate=torch.tensor(mask_result).resize(h*w)         
       
        # indices = torch.cat([torch.stack(list(torch.where(feat_immediate)),dim=0) for _ in range(b)],dim=0).cuda()
        num = indices.size()[1]
        #  0324 mask设置-*----------------------------------------------
        # indice =torch.cat(torch.from_numpy([np.argwhere(torch.ones(8,8).resize(8*8).unsqueeze(0).numpy()) for _ in range(b)]),dim=0)
        # feat_indices = np.argwhere(torch.cat([torch.ones(8,8).resize(8*8).unsqueeze(0) for _ in range(b)],
        #                          dim=0).numpy()>0)
        

        feat = feat.view(b, c, -1)  # (32*128*64)
        feat_select = batched_index_select(feat, dim=2, index=indices)  # (32*128*12)

        # local matching
        w_feat_select = batched_index_select(w_feat, dim=1, index=indices)  # (32*12*128)
        w_feat_select = F.normalize(w_feat_select, dim=2)  # (32*12*128)

        refs = refs.view(b, n, c, h * w)
        ref_indices = []
        ref_selects = []
        #  固定mask作为index
        ## 
        ###   固定  分割标签索引
        # for j in range(n):
        #     ref = refs[:, j, :, :]  # (32*128*64)
        #     ref_mask = mask[:, j, :, :]
        #     indices =batched_mask_stack(ref_mask,b,h,w,max_batch_length=max_index_length).cuda()
        #     # w_ref = w_refs.view(b, c, n, h * w)[:, :, j, :]  # (32*128*64)
        #     # fx = torch.matmul(w_feat_select, w_ref)  # (32*12*64)
        #     # _, indice = torch.topk(fx, dim=2, k=1,largest=False)
        #     # indice = indice.squeeze(0).squeeze(-1)  # (32*10)
        #     select = batched_index_select(ref, dim=2, index=indices)  # (32*128*12)
        #     ref_indices.append(indices)
        #     ref_selects.append(select)
        ###   固定  分割标签索引
        ## 原版带相似度
        for j in range(n):
            ref = refs[:, j, :, :]  # (32*128*64)
            w_ref = w_refs.view(b, c, n, h * w)[:, :, j, :]  # (32*128*64)
            fx = torch.matmul(w_feat_select, w_ref)  # (32*12*64)
            _, indice = torch.topk(fx, dim=2, k=1)
            indice = indice.squeeze(0).squeeze(-1)  # (32*10)
            select = batched_index_select(ref, dim=2, index=indice)  # (32*128*12)
            ref_indices.append(indice)
            ref_selects.append(select)
        ref_indices = torch.cat([item.unsqueeze(1) for item in ref_indices], dim=1)  # (32*2*12)
        ref_selects = torch.cat([item.unsqueeze(1) for item in ref_selects], dim=1)  # (32*2*128*12)

        # local replacement
        base_similarity = base_similarity.view(b, 1, 1)  # (32*1*1)
        ref_similarities = ref_similarities.view(b, 1, n)  # (32*1*2)
        feat_select = feat_select.view(b, 1, -1)  # (32*1*(128*12))
        ref_selects = ref_selects.view(b, n, -1)  # (32*2*(128*12))

        feature_fused = torch.cat((feat_select,ref_selects),dim=1) # (32*1+2*(128*12))
        feature_fused = feature_fused.view(b,-1,c)
        ###  原版本
        # feat_fused = torch.matmul(base_similarity, feat_select) \
        #              + torch.matmul(ref_similarities, ref_selects)  # (32*1*(128*12))
        # feat_fused = feat_fused.view(b, c, num)  # (32*128*12)

        # feat = batched_scatter(feat, dim=2, index=indices, src=feat_fused)
        # feat = feat.view(b,c,h*w)  # (32*128*8*8)
        ### 原版本
        RNN_out, (h_n, h_c) = self.LSTM(feature_fused, None)
        RNN_out = RNN_out[:,-1,:]
        
        return RNN_out, indices, ref_indices  # (32*128*8*8), (32*12), (32*2*12)

##### ----------------LocalFusionModule-------------------####

##### ----------------LocalFusionattentionModule-------------------####
class LocalFusionModule_mask_attention(nn.Module):
    def __init__(self, inplanes, rate,mask,CNN_embed_dim=128,h_RNN_layers=3, h_RNN=128):
        super(LocalFusionModule_mask_attention, self).__init__()

        
        self.W = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inplanes)
        )
        self.rate = rate
        self.mask = mask
        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN
        self.LSTM  = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,        
            num_layers=h_RNN_layers,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
    def forward(self, feat, refs, index, similarity,mask):
        
        refs = torch.cat([refs[:, :index, :, :, :], refs[:, (index + 1):, :, :, :]], dim=1)
        base_similarity = similarity[:, index]
        ref_similarities = torch.cat([similarity[:, :index], similarity[:, (index + 1):]], dim=1)

        # take ref:(32, 2, 128, 8, 8) for example
        b, n, c, h, w = refs.size()
        refs = refs.view(b * n, c, h, w)

        w_feat = feat.view(b, c, -1)
        w_feat = w_feat.permute(0, 2, 1).contiguous()
        w_feat = F.normalize(w_feat, dim=2)  # (32*64*128)

        w_refs = refs.view(b, n, c, -1)
        w_refs = w_refs.permute(0, 2, 1, 3).contiguous().view(b, c, -1)
        w_refs = F.normalize(w_refs, dim=1)  # (32*128*128)
        max_index_length = find_max_length(mask,b,n+1,h,w)
         #  0324 mask设置
        base_mask = mask[:, index, :, :, :]
        indices = batched_mask_stack(base_mask,b,h,w,max_index_length).cuda()
        num = indices.size()[1]

        feat = feat.view(b, c, -1)  # (32*128*64)
        feat_select = batched_index_select(feat, dim=2, index=indices)  # (32*128*12)

        # local matching
        w_feat_select = batched_index_select(w_feat, dim=1, index=indices)  # (32*12*128)
        w_feat_select = F.normalize(w_feat_select, dim=2)  # (32*12*128)

        refs = refs.view(b, n, c, h * w)
        ref_indices = []
        ref_selects = []
        #  固定mask作为index
        ## 
        # for j in range(n):
        #     ref = refs[:, j, :, :]  # (32*128*64)
        #     ref_mask = mask[:, j, :, :]

        #     indices =batched_mask_stack(ref_mask,b,h,w,max_batch_length=max_index_length).cuda()
        #     select = batched_index_select(ref, dim=2, index=indices)  # (32*128*12)
        #     ## 添加相似度选择
        #     ref_indices.append(indices)
        #     ref_selects.append(select)

       ##  原版索引
        for j in range(n):
            ref = refs[:, j, :, :]  # (32*128*64)
            w_ref = w_refs.view(b, c, n, h * w)[:, :, j, :]  # (32*128*64)
            fx = torch.matmul(w_feat_select, w_ref)  # (32*12*64)
            _, indice = torch.topk(fx, dim=2, k=1)
            indice = indice.squeeze(0).squeeze(-1)  # (32*10)
            select = batched_index_select(ref, dim=2, index=indice)  # (32*128*12)
            ref_indices.append(indice)
            ref_selects.append(select)


        ref_indices = torch.cat([item.unsqueeze(1) for item in ref_indices], dim=1)  # (32*2*12)
        ref_selects = torch.cat([item.unsqueeze(1) for item in ref_selects], dim=1)  # (32*2*128*12)


        # local replacement
        base_similarity = base_similarity.view(b, 1, 1)  # (32*1*1)
        ref_similarities = ref_similarities.view(b, 1, n)  # (32*1*2)
        feat_select = feat_select.view(b, 1, -1)  # (32*1*(128*12))
        ref_selects = ref_selects.view(b, n, -1)  # (32*2*(128*12))

        feature_fused = torch.cat((feat_select,ref_selects),dim=1) # (32*1+2*(128*12))
        feature_fused = feature_fused.view(b,-1,c)
        ###  原版本
        feat_fused = torch.matmul(base_similarity, feat_select) \
                     + torch.matmul(ref_similarities, ref_selects)  # (32*1*(128*12))
        feat_fused = feat_fused.view(b, c, num)  # (32*128*12)

        feat = batched_scatter(feat, dim=2, index=indices, src=feat_fused)
        feat = feat.view(b,c,h,w)  # (32*128*8*8)
        ### 原版本
        # RNN_out, (h_n, h_c) = self.LSTM(feature_fused, None)
        # RNN_out = RNN_out[:,-1,:]
        
        return feat, indices, ref_indices  # (32*128*8*8), (32*12), (32*2*12)

class LocalFusionModule_refindices(nn.Module):
    def __init__(self):
        super(LocalFusionModule_refindices, self).__init__()

    def forward(self, feat, refs, index, similarity,mask,device):
        
        refs = torch.cat([refs[:, :index, :, :, :], refs[:, (index + 1):, :, :, :]], dim=1)
        base_similarity = similarity[:, index]
        ref_similarities = torch.cat([similarity[:, :index], similarity[:, (index + 1):]], dim=1)
        ref_masks = torch.cat([mask[:, :index, :, :, :], mask[:, (index + 1):, :, :, :]], dim=1)
        # take ref:(32, 2, 128, 8, 8) for example
        b, n, c, h, w = refs.size()
        refs = refs.view(b * n, c, h, w)

        w_feat = feat.view(b, c, -1)
        w_feat = w_feat.permute(0, 2, 1).contiguous()
        w_feat = F.normalize(w_feat, dim=2)  # (32*64*128)

        w_refs = refs.view(b, n, c, -1)
        w_refs = w_refs.permute(0, 2, 1, 3).contiguous().view(b, c, -1)
        w_refs = F.normalize(w_refs, dim=1)  # (32*128*128)

        max_index_length = find_max_length(mask,b,n+1,h,w)
       
        base_mask = mask[:, index, :, :, :]
        indices = batched_base_mask_stack(base_mask,b,h,w,max_index_length).cuda(device)

        num = indices.size()[1]

        feat = feat.view(b, c, -1)  # (32*128*64)
        feat_select = batched_index_select(feat, dim=2, index=indices)  # (32*128*12)

        # local matching
        w_feat_select = batched_index_select(w_feat, dim=1, index=indices)  # (32*12*128)
        w_feat_select = F.normalize(w_feat_select, dim=2)  # (32*12*128)

        refs = refs.view(b, n, c, h * w)
        ref_indices = []
        ref_selects = []
     
        for j in range(n):
            ref = refs[:, j, :, :]  # (32*128*64)

            ref_mask = ref_masks[:, j, :, :, :]
            ref_indice = batched_base_mask_stack(ref_mask,b,h,w,max_index_length).cuda(device)

            w_ref = w_refs.view(b, c, n, h * w)[:, :, j, :]  # (32*128*64)
            w_ref_feat_select = batched_index_select(w_ref, dim=2, index=ref_indice)  # (32*12*128) 利用 ref_indice 进行初筛
            # w_ref_feat_select = F.normalize(w_ref_feat_select, dim=2)  # (32*12*128)
        
            fx = torch.matmul(w_feat_select, w_ref_feat_select)  # (32*12*64)
            _, indice = torch.topk(fx, dim=2, k=1)
            indice = indice.squeeze(0).squeeze(-1)  # (32*10)
            select = batched_index_select(w_ref_feat_select, dim=2, index=indice)  # (32*128*12)
            ref_indices.append(indice)
            ref_selects.append(select)
        ref_indices = torch.cat([item.unsqueeze(1) for item in ref_indices], dim=1)  # (32*2*12)
        ref_selects = torch.cat([item.unsqueeze(1) for item in ref_selects], dim=1)  # (32*2*128*12)

        # local replacement
        base_similarity = base_similarity.view(b, 1, 1)  # (32*1*1)
        ref_similarities = ref_similarities.view(b, 1, n)  # (32*1*2)
        feat_select = feat_select.view(b, 1, -1)  # (32*1*(128*12))
        ref_selects = ref_selects.view(b, n, -1)  # (32*2*(128*12))

        feature_fused = torch.cat((feat_select,ref_selects),dim=1) # (32*1+2*(128*12))
        feature_fused = feature_fused.view(b,-1,c)
        ###  原版本
        feat_fused = torch.matmul(base_similarity, feat_select) \
                     + torch.matmul(ref_similarities, ref_selects)  # (32*1*(128*12))
        feat_fused = feat_fused.view(b, c, num)  # (32*128*12)

        feat = batched_scatter(feat, dim=2, index=indices, src=feat_fused)
        feat = feat.view(b,c,h,w)  # (32*128*8*8)

        return feat, indices, ref_indices  # (32*128*8*8), (32*12), (32*2*12)


class LocalFusionModule_featattention_refindices(nn.Module):
    def __init__(self):
        super(LocalFusionModule_featattention_refindices, self).__init__()     
        self.w0 = nn.Parameter(torch.ones(2))
        self.fusion = LocalFusionModule_refindices()
        # self.fusion = LocalFusionModule_base_mask_lstm(inplanes=128, rate=None,mask=None)
    def forward(self, xs_immed,similarity,mask,device):
        # querys1=xs_immed.clone()
        media_query =[]
        k = mask.size()[1]
        if k==5:
            for base_index in range(k):
                
                    base_feat = xs_immed[:, base_index, :, :, :]
                
                    feat_gen, indices_feat, indices_ref= self.fusion(base_feat, xs_immed, base_index, similarity,mask,device)
                    media_query.append(feat_gen)

            feat_gen=torch.stack(media_query,dim=1)
            w1 = torch.exp(self.w0[0])/torch.sum(torch.exp(self.w0))
            w2 = torch.exp(self.w0[1])/torch.sum(torch.exp(self.w0))
            querys1=w1*xs_immed+w2*feat_gen
            # xs= xs_immed.clone().to(torch.float32)*torch.tensor(0.5,dtype=torch.float32)
            return querys1
            # return xs


class  changemask(nn.Module):
    def __init__(self):
        super(changemask, self).__init__()
        
        self.pool=nn.AdaptiveAvgPool3d(3)
        self.softmax = nn.Softmax(dim =2)
        self.sigmoid =nn.Sigmoid()
        self.w0 = nn.Parameter(torch.ones(2))
    def forward(self, feat, mask,device):
        feat = torch.sum(feat,dim=2).unsqueeze(2)
        # feat = self.softmax(feat)
        feat = self.sigmoid(feat)
        w1 = torch.exp(self.w0[0])/torch.sum(torch.exp(self.w0))
        w2 = torch.exp(self.w0[1])/torch.sum(torch.exp(self.w0))
        mask =w1*mask+w2*feat
        # mask = mask
        return mask

class resnet18_fpn_branch_wrefindices_changemask_attention(nn.Module):
    def __init__(self, num_classes,mask,device):
        super(resnet18_fpn_branch_wrefindices_changemask_attention, self).__init__()
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
        
        self.toplayer = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)  
        ##  smoth
        self.smooth1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.latlayer1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.latlayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.latlayer3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        
        n_class = num_classes
        self.pool = nn.AdaptiveAvgPool2d(1)
             
        self.linear = nn.Linear(4*n_class,n_class)
        self.LSTM  = nn.LSTM(
            input_size=512,
            hidden_size=256,        
            num_layers=1,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.featmap_LSTM128  = nn.LSTM(
            input_size=128,
            hidden_size=128,        
            num_layers=1,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.featmap_LSTM256  = nn.LSTM(
            input_size=256,
            hidden_size=128,        
            num_layers=1,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.featmap_LSTM512  = nn.LSTM(
            input_size=512,
            hidden_size=128,        
            num_layers=1,       
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.fc1 = nn.Linear(128, 13)
        self.fc2 = nn.Linear(13, n_class)
        # self.fc2 = nn.Linear(8, n_class)
        # self.fc2 = nn.Linear(16, n_class)
        self.featmap_fc128 = nn.Linear(128,64)
        self.featmap_class = nn.Linear(64,3)
        
        self.featmap_fc256 = nn.Linear(256,3)
        
        cnn_c = [nn.AdaptiveAvgPool2d(1),
                 Conv2dBlock(512, n_class, 1, 1,
                             norm='none',
                             activation='none',
                             activation_first=False)]
        self.cnn_c = nn.Sequential(*cnn_c)
        

        # self.fusion0 = LocalFusionModule_refindices()
        self.one_stage_feat_attention = LocalFusionModule_featattention_refindices()
        self.two_stage_feat_attention = LocalFusionModule_featattention_refindices()
        self.three_stage_feat_attention = LocalFusionModule_featattention_refindices()
        self.four_stage_feat_attention = LocalFusionModule_featattention_refindices()
        self.five_stage_feat_attention = LocalFusionModule_featattention_refindices()
        
        self.fpn_stage_feat_attention = LocalFusionModule_featattention_refindices()
        self.w_feat = nn.Parameter(torch.ones(3))  
        self.changemask1 = changemask()
        self.changemask2 = changemask()
        self.changemask3 = changemask()
        self.changemask4 = changemask()
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
            
        )
        # similarity = torch.cat([torch.ones(8,1) for _ in range(5)],dim=1).cuda(device)
        # similarity_total = torch.cat([torch.ones(8,1) for _ in range(5)],dim=1).cuda(device)
        # similarity_sum = torch.sum(similarity_total, dim=1, keepdim=True).expand(8, 5)
        # self.similarity = torch.autograd.variable( similarity_total / similarity_sum)  # b*k
        self.fc_loc=nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(True),
            nn.Linear(16,6)
        )
        #注意注意！！！！STN依赖参数初始化
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))  #STN module的参数初始化非常条件，0初始化或者选择高斯初始化，方差极低
        
        self.drop_out = nn.Dropout(0.5)
      
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def stn(self,x):
        xs=self.localization(x)
        xs=xs.view(-1,32)
        theta=self.fc_loc(xs)
        theta=theta.view(-1,2,3)
        grid=F.affine_grid(theta,x.size())   #构建STN自适应仿射变换
        x=F.grid_sample(x,grid)
        return x
    
    def forward(self, xs,mask):
        ### 0523 集成化
           # Bottom-up
        b, k, C, H, W = xs.size()
        device = self.device
        ## 初始相似性  
        similarity_total = torch.cat([torch.ones(b,1) for _ in range(k)],dim=1).cuda(device)
        similarity_sum = torch.sum(similarity_total, dim=1, keepdim=True).expand(b, k)
        self.similarity = similarity_total / similarity_sum  # b*k

        b, k, m_C, H,W = mask.size()     
        mask_orginal = mask.view(-1,m_C,H,W)

        xs = self.one_stage_feat_attention(xs,self.similarity,mask,device)

        ###  第一阶段
        xs = xs.view(-1, C, H, W)
        querys = self.resnet_head(xs)         
        c, h, w = querys.size()[-3:]
        querys = querys.view(b, k, c, h, w)      
        ### 测试是不是插值
        # mask = self.mask_head(mask)
        mask = interpolate(mask_orginal,size=[h,w])  # 第一阶段插值
        mask = mask.view(b, k, m_C, h, w)      
        

        querys1 =  self.two_stage_feat_attention(querys,self.similarity,mask,device)
        mask_1 =self.changemask1(querys1,mask,device)

        ###  第二阶段
        querys1 = self.conv2d_1(querys1.view(-1,c,h,w))
        mask_1 = self.mask_conv2d_1(mask_1.view(-1,m_C,h,w))       
        c1, h1, w1 = querys1.size()[-3:]
        querys1 = querys1.view(b, k, c1, h1, w1)
        mask_1 = mask_1.view(b, k,m_C , h1, w1)
        mask_interp = interpolate(mask_orginal,size=[h1,w1]).view(b, k,m_C , h1, w1)
        mask= mask_interp+mask_1
        # mask = mask_2
        # querys2=querys1.clone()
       
        querys2 =  self.three_stage_feat_attention(querys1,self.similarity,mask_1,device)    
        mask_2 =self.changemask2(querys2,mask_1,device)

        ###  第三阶段
        querys2 = self.conv2d_2(querys2.view(-1,c1,h1,w1))
        # querys3 =querys2.clone()
        mask_2 = self.mask_conv2d_2(mask_2.view(-1,m_C,h1,w1))

        c2, h2, w2 = querys2.size()[-3:]
        querys2 = querys2.view(b, k, c2, h2, w2)
        mask_2 = mask_2.view(b, k,m_C , h2, w2)
        mask_interp = interpolate(mask_orginal,size=[h2,w2]).view(b, k,m_C , h2, w2)
        mask= mask_interp+mask_2
        # mask =mask_2
        querys3 =  self.four_stage_feat_attention(querys2,self.similarity,mask_2,device)   
        
        mask_3 =self.changemask3(querys3,mask_2,device)

        ### 第四阶段
        querys3 = self.conv2d_3(querys3.view(-1,c2,h2,w2))
        mask_3 = self.mask_conv2d_3(mask_3.view(-1,m_C,h2,w2))
        ## 0402 转换成函数
        c3, h3, w3 = querys3.size()[-3:]
        querys3 = querys3.view(b, k, c3, h3, w3)
        mask_3 = mask_3.view(b, k,m_C , h3, w3)
        mask_interp = interpolate(mask_orginal,size=[h3,w3]).view(b, k,m_C , h3, w3)
        mask= mask_interp+mask_3
        # mask =mask_2
        # querys4=querys3.clone()
        # similarity = torch.cat([torch.ones(b,1) for _ in range(k)],dim=1).cuda(device)
        querys4 =  self.five_stage_feat_attention(querys3,self.similarity,mask_3,device)   
        # querys4 = querys4.view(-1,c3,h3,w3)
        mask_4 =self.changemask4(querys4,mask_3,device)
        # p4=self.toplayer(querys4)
        # p4 = self._upsample_add(p4,querys3.view(-1,c3,h3,w3))
        # p4 = self.smooth1(p4)
        # p3 = self._upsample_add(p4,querys2.view(-1,c2,h2,w2))
        # p3 =self.smooth2(p3)
        # p2 =self._upsample_add(p3,querys1.view(-1,c1,h1,w1))
        # p2 = self.smooth3(p2)
        # Top-down
        # feature_maps = [p2.view(b,-1,c1,h1,w1)]#, p3.view(b,-1,c1,h2,w2), p4.view(b,-1,c2,h3,w3)]
        feature_maps =[ querys4,querys3,querys2]
        # feature_maps =[ querys4]# 超参敏感性实验
        # feature_maps =[ querys4,querys3]# 超参敏感性实验
        
        w1 = torch.exp(self.w_feat[0])/torch.sum(torch.exp(self.w_feat))
        w2 = torch.exp(self.w_feat[1])/torch.sum(torch.exp(self.w_feat))
        w3 = torch.exp(self.w_feat[2])/torch.sum(torch.exp(self.w_feat))
        masks_maps = [mask_3,mask_4,mask_1,mask_2]
        w_feature_maps =[w1,w2,w3]
        mask_all =[]
        MASK_H,MASK_W = mask.size()[-2:]
        n_mask_map =len(masks_maps)
        for i  in range(n_mask_map):

            mask_map = masks_maps[i]
            c,h,w = mask_map.size()[-3:]
           
            mask = self._upsample_add(mask_map.view(-1,c,h,w),mask.view(-1,c,MASK_H,MASK_W))

        n_feat_map =len(feature_maps)

       #### classfier
        class_score_all =[]
        n_feat_map =len(feature_maps)
        for  i  in range(n_feat_map):
            feat_map =feature_maps[i]
            c,h,w = feat_map.size()[-3:]
            mask_immediate = interpolate(mask,size=[h,w])
            class_score =  self.fpn_stage_feat_attention(feat_map,self.similarity,mask_immediate.view(b,-1,m_C,h,w),device)   
            ## grad cam时 注意unsqueeze（） batch 纬度
            class_score=self.pool(feat_map.view(b,-1,c,h,w)).squeeze()#.unsqueeze(0)

            if c == 128:
                class_predict ,(h_n, h_c) =self.featmap_LSTM128(class_score,None)
            if c == 256:
                class_predict,(h_n, h_c)  =self.featmap_LSTM256(class_score,None)
            if c== 512:
               class_predict,(h_n, h_c)  =self.featmap_LSTM512(class_score,None)
    #         else:
    #             assert  "fc_layer not 128 and 256"
            class_predict =class_predict[:,-1,]
            
            class_score = self.featmap_fc128(class_predict)
            class_score = F.relu(class_score)
            class_score=self.drop_out(class_score)
            class_score = self.featmap_class(class_score)*w_feature_maps[i]
            class_score_all.append(class_score)
            
        class_score_alls = sum(class_score_all)
        # RNN_out, (h_n, h_c) = self.featmap_LSTM128(class_score_alls, None)
    #0526  

        # querys4 = self.pool(querys4.view(b,-1,c3,h3,w3)).squeeze()

        # mask = self.pool(mask).squeeze()
        # ## 0402 转换成函数
        # RNN_out, (h_n, h_c) = self.featmap_LSTM(querys4, None)
        # # RNN_out = querys4
        # x= RNN_out[:,-1,]
        # x = self.fc1(x)
        # x = F.relu(x)
        # x=self.drop_out(x)
        # # x= torch.concat((x,mask),dim=1)
        # x = self.fc2(x)
        # return x
        return class_score_alls
      