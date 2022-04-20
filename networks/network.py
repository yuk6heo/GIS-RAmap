import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from networks.deeplab.aspp import build_aspp, ASPP

from networks.deeplab.backbone.resnet import SEResNet50


class NET_GAmap(nn.Module):
    # with Indicator encoder 
    def __init__(self, pretrained=1, resfix=False,):
        super(NET_GAmap, self).__init__()

        # Sparse-to-Dense Network & Object Feature Extractor
        self.encoder_6ch = Encoder_6ch(resfix)
        self.decoder_iact = Decoder()
        self.converter = Object_Feature_Extractor()

        # Interfused Object Feature
        self.encoder_3ch = Encoder_3ch(resfix)
        self.transfer_module = Attention_Transfer_Module()

        # Overlapped Object Feature
        self.diff_module = IAware_Diff_Prop_Module()

        # Segmentation Head
        self.feature_compounder = Feature_compounder()
        self.segmentation_head = Segmentation_Head()

        self.refer_weight = None
        self._initialize_weights(pretrained)

    def is_training(self,boolean):
        if boolean == True:
            self.transfer_module.training = True
            self.diff_module.training = True
        else:
            self.transfer_module.training = False
            self.diff_module.training = False

    def forward_obj_feature_extractor(self, x): # Bx4xHxW to Bx1xHxW
        r5, _, r3, r2 = self.encoder_6ch(x)
        estimated_mask, m2 = self.decoder_iact(r5, r3, r2, train_prop=False)
        r5_indicator = self.converter(r5, r3, m2)
        return estimated_mask, r5_indicator

    def forward_prop(self, anno_propEnc_r4_list, queframe_3ch, anno_iactEnc_r4_list, r4_neighbor, neighbor_pred_onehot,
                     anno_fr_list=None, que_fr=None, debug=False): #1/16, 1024
        if debug == False:
            r4_que, r2_que = self.encoder_3ch(queframe_3ch)
            trf_module_out, scoremap = self.transfer_module(anno_propEnc_r4_list, r4_que, anno_iactEnc_r4_list, anno_fr_list, que_fr) # 1/8, 256
            diff_module_out = self.diff_module(neighbor_pred_onehot, r4_neighbor, r4_que)
            m2 = self.feature_compounder(trf_module_out, diff_module_out, r4_que, r2_que)
            estimated_fgbg = self.segmentation_head(m2)

            fg_map = (F.softmax(F.interpolate(estimated_fgbg, scale_factor=0.125), dim=0)[:,1] > 0.4).float() # Nobj,H,W
            fg_map = torch.max(fg_map, dim=0)[0] #H W
            n_fg = fg_map.sum()
            score = (float(torch.mean(scoremap) + (torch.sum(fg_map * scoremap)/(n_fg+0.1)).cpu()))/2

            return estimated_fgbg, r4_que, score

        else:
            r4_que, r2_que = self.encoder_3ch(queframe_3ch)
            trf_module_out, scoremap, attention = self.transfer_module(anno_propEnc_r4_list, r4_que, anno_iactEnc_r4_list, anno_fr_list, que_fr, True) # 1/8, 256
            diff_module_out = self.diff_module(neighbor_pred_onehot, r4_neighbor, r4_que)
            m2 = self.feature_compounder(trf_module_out, diff_module_out, r4_que, r2_que)
            estimated_fgbg = self.segmentation_head(m2)

            fg_map = (F.softmax(F.interpolate(estimated_fgbg, scale_factor=0.125), dim=0)[:,1] > 0.4).float() # Nobj,H,W
            fg_map = torch.max(fg_map, dim=0)[0] #H W
            n_fg = fg_map.sum()
            score = (float(torch.mean(scoremap) + (torch.sum(fg_map * scoremap)/(n_fg+0.1)).cpu()))/2

            return estimated_fgbg, r4_que, score, attention


    def _initialize_weights(self, pretrained):
        for m in self.modules():
            if pretrained:
                break
            else:
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0, 0.001)
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Encoder_3ch(nn.Module):
    def __init__(self, resfix):
        super(Encoder_3ch, self).__init__()

        self.conv0_3ch = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)

        resnet = SEResNet50(output_stride=8, BatchNorm=nn.BatchNorm2d, pretrained=True)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/8, 1024

        # freeze BNs
        if resfix:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, x):
        # x : [b,4,h,w]
        # f = (in_frame - Variable(self.mean)) / Variable(self.std)
        #a = torch.unsqueeze(in_a, dim=1).float()  # add channel dim
        #b = torch.unsqueeze(in_b, dim=1).float()  # add channel dim

        x = self.conv0_3ch(x)  # 1/2, 64
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/8, 1024

        return r4, r2


class Encoder_6ch(nn.Module):
    def __init__(self, resfix):
        super(Encoder_6ch, self).__init__()

        self.conv0_6ch = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=True)

        resnet = SEResNet50(output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024
        self.res5 = resnet.layer4  # 1/16, 2048

        # freeze BNs
        if resfix:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters():
                        p.requires_grad = False

    def forward(self, x):
        # x : [b,4,h,w]
        # f = (in_frame - Variable(self.mean)) / Variable(self.std)
        #a = torch.unsqueeze(in_a, dim=1).float()  # add channel dim
        #b = torch.unsqueeze(in_b, dim=1).float()  # add channel dim

        x = self.conv0_6ch(x)  # 1/2, 64
        x = self.bn1(x)
        c1 = self.relu(x)  # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)  # 1/4, 256
        r3 = self.res3(r2)  # 1/8, 512
        r4 = self.res4(r3)  # 1/16, 1024
        r5 = self.res5(r4)  # 1/16, 2048

        return r5, r4, r3, r2



class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.convFS2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convFS3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.convMM2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.convFS1(f)
        sr = self.convFS2(F.relu(s))
        sr = self.convFS3(F.relu(sr))
        s = s + sr

        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode='bilinear',align_corners=True)
        mr = self.convMM1(F.relu(m))
        mr = self.convMM2(F.relu(mr))
        m = m + mr
        return m


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        mdim = 256

        self.aspp_decoder = ASPP(backbone='res', output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=1)
        self.convG0 = nn.Conv2d(2048, mdim, kernel_size=3, padding=1)
        self.convG1 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.convG2 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)

        self.RF3 = Refine(512, mdim)  # 1/16 -> 1/8
        self.RF2 = Refine(256, mdim)  # 1/8 -> 1/4

        self.lastconv = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.Conv2d(256, 1, kernel_size=1, stride=1))

    def forward(self, r5, que_r3, que_r2, train_prop = True):

        aspp_out = self.aspp_decoder(r5) #1/16 mdim
        aspp_out = F.interpolate(aspp_out, scale_factor=4, mode='bilinear',align_corners=True) #1/4 mdim
        m4 = self.convG0(F.relu(r5))  # out: # 1/16, mdim
        m4 = self.convG1(F.relu(m4))  # out: # 1/16, mdim
        m4 = self.convG2(F.relu(m4)) # out: # 1/16, mdim


        m3 = self.RF3(que_r3, m4)  # out: 1/8, mdim
        m2 = self.RF2(que_r2, m3)  # out: 1/4, mdim
        m2 = torch.cat((m2, aspp_out), dim=1) # out: 1/4, mdim*2

        if train_prop:
            return m2
        else:
            x = self.lastconv(m2)
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

            return x, m2



class IAware_Diff_Prop_Module(nn.Module):
    def __init__(self):
        super(IAware_Diff_Prop_Module, self).__init__()
        self.f_conv_inter = nn.Conv2d(1024, 128, kernel_size=1, padding=0)  # 1/8, 128
        self.conv1_inter = nn.Conv2d(128, 64, kernel_size=5, padding=2)  # 1/8, 128

        self.f_conv_diffu = nn.Conv2d(1024, 128, kernel_size=1, padding=0)  # 1/8, 128
        self.conv1_diffu = nn.Conv2d(129, 64, kernel_size=5, padding=2)  # 1/8, 128
        self.conv2_diffu = nn.Conv2d(128, 128, kernel_size=5, padding=2)  # 1/8, 128
        self.conv3_diffu = nn.Conv2d(128, 128, kernel_size=5, padding=2)  # 1/8, 128
        self.training = True # If testing, batchsize = n_obj

    def forward(self, neighbor_pred_onehot, r4_nei, r4_que):
        '''
        neighbor_pred_onehot:
        Train: # B,1,H,W
        Test: # Nobj,1,H,W
        '''
        f_inter = (self.f_conv_inter(r4_nei) - self.f_conv_inter(r4_que))**2 # [B, C, 128, HW]
        f_inter = self.conv1_inter(torch.exp(-f_inter))

        neighbor_pred_onehot = F.interpolate(neighbor_pred_onehot, scale_factor=0.125, mode='bilinear',align_corners=True) #1/4 mdim
        f_diff = torch.cat((self.f_conv_diffu(r4_nei), neighbor_pred_onehot), dim=1)
        f_diff = self.conv1_diffu(f_diff)
        f_diff = torch.cat((f_diff,f_inter), dim=1)
        f_diff = self.conv2_diffu(f_diff)
        f_diff = self.conv3_diffu(f_diff)

        return f_diff


class Attention_Transfer_Module(nn.Module):
    def __init__(self):
        super(Attention_Transfer_Module, self).__init__()
        self.f_conv_sim = nn.Conv2d(1024, 128, kernel_size=1, padding=0)  # 1/8, 128
        self.f_conv_att = nn.Conv1d(128, 128, kernel_size=1, padding=0)  # 1/8, 128
        self.training = True # If testing, batchsize = n_obj

    def get_attention(self, anno_f, que_f, similarity_mat, h, w):
        '''
        anno_f :           train [BN, 128, HW']   test [N, 128, HW']
        que_f :            train [B, 128, HW]       test [1, 128, HW]
        similarity_mat :   train [BN, HW', HW]      test [N, HW', HW]
        '''
        bn, c, hw = anno_f.size()
        b, c, hw = que_f.size()
        n_feature = int(bn/b)

        similarity_mat_self = F.softmax(torch.bmm(que_f.transpose(1, 2), que_f), dim=2)
        que_f_transferred = torch.bmm(self.f_conv_att(que_f), similarity_mat_self).unsqueeze(dim=1) # [B, 1, 128, HW]
        anno_f_transferred = torch.bmm(self.f_conv_att(anno_f), similarity_mat).reshape(b, n_feature, c, hw) # [B, N, 128, HW]

        # diff = (anno_f_transferred - que_f_transferred)**2 # [B, N, 128, HW]
        # attention = (torch.max(diff, dim=2, keepdim=True)[0]).reshape(b,n_feature,1,h,w) # [B, N, 1, H, W]
        # attention = F.softmax(1/(attention+0.1),dim=1) # [B, N, 1, H, W]
        diff = (anno_f_transferred - que_f_transferred)**2 # [B, N, 128, HW]
        diff = (torch.max(diff, dim=2, keepdim=True)[0]).reshape(b,n_feature,1,h,w) + 0.1 # [B, N, 1, H, W]
        attention_logit = 1/diff
        scoremap_logit = attention_logit.clone() # [B, N, 1, H, W]
        scoremap_logit = scoremap_logit[0,:,0 ] # [N, H, W]
        scoremap = torch.exp(torch.max(scoremap_logit, dim=0)[0]/2-5) # H, W
        attention = F.softmax(attention_logit,dim=1) # [B, N, 1, H, W]
        return attention, scoremap

    def forward(self, anno_feature_list, que_feature, anno_indicator_feature_list, anno_fr_list, que_fr, debug = False):
        '''

        :param anno_feature_list: [B,C,H,W] x list (N values in list) B-Nobject N-Nround
        :param que_feature:  [B,C,H,W]
        :param anno_indicator_feature_list:  [B,C,H,W] x list (N values in list)
        :return que_mask_feature: [B,C,H,W]
        '''

        n_features = len(anno_feature_list)
        b, ci, h, w = anno_indicator_feature_list[0].size() # b means n_objs # [B, 256, HxW]

        if (n_features >= 4) and (anno_fr_list is not None):
            anno_fr_list_tmp = anno_fr_list[1:]
            index_adjacent = np.argsort(np.abs(que_fr - anno_fr_list_tmp))[:2]
            anno_fr_list = anno_fr_list[0] + list(anno_fr_list_tmp[index_adjacent])
            anno_fr_adjacent_index = [0] + list(1 + index_adjacent)
            n_features = 3
        else:
            anno_fr_adjacent_index = list(range(len(anno_feature_list)))


        anno_feature_sim = [] # [BN, C, HW'](train), # [N, C, HW'](test)
        anno_indicator_feature = [] # [BN, C, HW']

        for f_idx in anno_fr_adjacent_index:
            anno_feature_sim.append(self.f_conv_sim(anno_feature_list[f_idx]).reshape(b, 128, h*w)) # [B, 128, HW']
            anno_indicator_feature.append(anno_indicator_feature_list[f_idx].reshape(b, 256, h*w)) # [B, 256, HW']

        que_feature_sim = self.f_conv_sim(que_feature).reshape(b, 128, h*w) # [B, 128, HW]
        anno_feature_sim = torch.stack(anno_feature_sim, dim=1) # [B, N, 128, HW']
        anno_indicator_feature = torch.stack(anno_indicator_feature, dim=1)  # [B, N, 256, HW']


        if self.training:
            anno_feature_sim = anno_feature_sim.reshape(b*n_features, 128, h*w) # [BN, 128, HW']
            que_feature_sim_tmp = torch.unsqueeze(que_feature_sim,dim=1).expand(-1, n_features, -1, -1).reshape(b*n_features, 128, h*w) # [BN, 128, HW']
            similarity_mat = F.softmax(torch.bmm(anno_feature_sim.transpose(1, 2), que_feature_sim_tmp), dim=2) # [BN, HW', HW]
            attention, scoremap = self.get_attention(anno_feature_sim, que_feature_sim, similarity_mat, h, w) # [B, N, 1, H, W]
            anno_indicator_feature = anno_indicator_feature.reshape(b*n_features, 256, h*w) # [B, N, 256, H, W]
            anno_indicator_feature_transferred = torch.bmm(anno_indicator_feature,similarity_mat).reshape(b, n_features, 256, h, w) # [B, N, 256, H, W]

        else:
            que_feature_sim = que_feature_sim[0]
            anno_feature_sim = anno_feature_sim[0].reshape(n_features, 128, h * w) # [N, 128, HW']
            que_feature_sim_tmp = torch.unsqueeze(que_feature_sim,dim=0).expand(n_features, -1, -1) # [N, 128, HW']
            similarity_mat = F.softmax(torch.bmm(anno_feature_sim.transpose(1, 2), que_feature_sim_tmp), dim=2) # [N, HW', HW]
            attention, scoremap = self.get_attention(anno_feature_sim, que_feature_sim.unsqueeze(dim=0), similarity_mat, h, w) # [1, N, 1, H, W]
            anno_indicator_feature_transferred = []
            for obj_idx in range(b):
                anno_indicator_feature_transferred.append(torch.bmm(anno_indicator_feature[obj_idx],similarity_mat))
            anno_indicator_feature_transferred = torch.stack(anno_indicator_feature_transferred,dim=0).reshape(b, n_features, 256, h, w) # [B, N, 256, H, W]

        que_mask_feature = torch.sum(anno_indicator_feature_transferred * attention, dim=1, keepdim=False) # [B, 256, H, W]

        if debug:
            return que_mask_feature, scoremap, attention.detach().data[0, :, 0].cpu().numpy()
        else:
            return que_mask_feature, scoremap


class Feature_compounder(nn.Module):
    def __init__(self):
        super(Feature_compounder, self).__init__()
        mdim = 128

        self.que_conv = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  # 1/8, 256
        self.cat_conv = nn.Conv2d(640, 512, kernel_size=1, padding=0)  # 1/8, 256

        self.aspp_decoder = ASPP(backbone='res', output_stride=8, BatchNorm=nn.BatchNorm2d, pretrained=1, inplanes=512, outplanes = mdim)
        self.convG0 = nn.Conv2d(512, mdim, kernel_size=3, padding=1)
        self.convG1 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)
        self.convG2 = nn.Conv2d(mdim, mdim, kernel_size=3, padding=1)

        self.RF2 = Refine(256, mdim)  # 1/8 -> 1/4

    def forward(self, trf_module_out, diff_module_out, que_r4, que_r2):

        que_r4 = self.que_conv(que_r4)
        r4 = torch.cat((trf_module_out, diff_module_out, que_r4), dim=1)
        r4 = self.cat_conv(r4)
        aspp_out = self.aspp_decoder(r4) #1/8 mdim
        aspp_out = F.interpolate(aspp_out, scale_factor=2, mode='bilinear',align_corners=True) #1/4 mdim
        m4 = self.convG0(F.relu(r4))  # out: # 1/8, mdim
        m4 = self.convG1(F.relu(m4))  # out: # 1/8, mdim
        m4 = self.convG2(F.relu(m4)) # out: # 1/8, mdim

        m2 = self.RF2(que_r2, m4)  # out: 1/4, mdim
        m2 = torch.cat((m2, aspp_out), dim=1) # out: 1/4, mdim*2

        return m2 # out: 1/4, 256

class Segmentation_Head(nn.Module):
    def __init__(self):
        super(Segmentation_Head, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 2, kernel_size=1, stride=1)

    def forward(self, seg_feature):
        '''

        :return:
        '''
        x = self.bn1(self.conv1(seg_feature))
        x = nn.Dropout(0.5)(F.relu(x))
        x = self.bn2(self.conv2(x))
        x = nn.Dropout(0.1)(F.relu(x))
        x = self.bn3(self.conv3(x))
        x = nn.Dropout(0.1)(F.relu(x))
        x = self.bn4(self.conv4(x))
        x = self.conv5(F.relu(x))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return x

class Object_Feature_Extractor(nn.Module):
    def __init__(self):
        super(Object_Feature_Extractor, self).__init__()
        # [1/4, 512] to [1/8, 256]
        downsample1 = nn.Conv2d(512, 256, kernel_size=1, stride=2, bias=False)
        self.block1 = SEBottleneck(512, 64, stride = 2, downsample = downsample1)

        # [1/16, 2048] to [1/8, 256]
        self.conv16_8 = nn.Conv2d(2048, 256, kernel_size=1, stride=1)

        # [1/8, 512] to [1/8, 256]
        self.conv8_8 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.conv_cat = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1)  # 1/8, 256

    def forward(self, r5, r4, m2):
        '''
        :param r5: 1/16, 2048
        :param r4: 1/8, 1024
        :param m2: 1/4, 512
        :return:
        '''
        m4 = self.block1(m2)
        r5 = self.conv16_8(r5)
        r5_r4 = F.interpolate(r5, scale_factor=2, mode='bilinear',align_corners=True)
        r4 = self.conv8_8(r4)
        x = torch.cat((r5_r4, r4, m4),dim=1)
        x = self.conv_cat(x)

        return x # 1/8, 256




class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=nn.BatchNorm2d):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        # SE
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            planes * 4, planes // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            planes // 4, planes * 4, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)

        if self.downsample is not None:
            residual = self.downsample(x)

        res = out1 * out + residual
        res = self.relu(res)

        return res
#
#
# if __name__ == "__main__":
#     import torch
#     model = ATnet()
#     input = torch.rand(1, 3, 512, 512)
#     output, low_level_feat = model(input)
#     print(output.size())
#     print(low_level_feat.size())