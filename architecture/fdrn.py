import torch
import torch.nn as nn
import model.blocks as blocks
import torch.nn.functional as F

def make_model(args):
    return FDRN(args)

'''Frequency Division Residual Network'''
class FDRN(nn.Module):
    def __init__(self,args):
        super(FDRN, self).__init__()

        n_feats = 64
        kernel_size = 3
        self.scale = args.scale[0]
        act = nn.ReLU(True)
        self.sub_mean = blocks.MeanShift(args.rgb_range)
        self.add_mean = blocks.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [blocks.conv_layer(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            FDRM() for _ in range(1)
        ]

        # define tail module
        m_tail = [
            blocks.Upsampler(n_feats,3,self.scale)
        ]



        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)


    def forward(self, x):
        x = self.sub_mean(x)
        x_shallow = self.head(x)

        x_deep = self.body(x_shallow)
        x_deep += x_shallow

        x_tail = self.tail(x_deep)
        x_inter = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.add_mean(x_tail+x_inter)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


'''Frequency Division Residual Module'''

class FDRM(nn.Module):
    def __init__(self, nc=64, kernel_size=1, n_dc = 4, n_residual = 2, res_scale=1):
        super(FDRM, self).__init__()
        self.dc1 = FDRG(nf=64, kernel_size=kernel_size, n_fd = n_dc, n_residual = n_residual,  res_scale=res_scale)
        self.dc2 = FDRG(nf=56, kernel_size=kernel_size, n_fd = n_dc, n_residual = n_residual,  res_scale=res_scale)
        self.dc3 = FDRG(nf=48, kernel_size=kernel_size, n_fd = n_dc, n_residual = n_residual,  res_scale=res_scale)
        self.dc4 = FDRG(nf=40, kernel_size=kernel_size, n_fd = n_dc, n_residual = n_residual,  res_scale=res_scale)
        self.dc5 = FDRG(nf=32, kernel_size=kernel_size, n_fd = n_dc, n_residual = n_residual,  res_scale=res_scale)
        self.conv1 = blocks.conv_layer(96,64,1)

    def forward(self, input):
        _,C,_,_ = input.shape
        out_dc1 = self.dc1(input)
        out_dc1, res_dc1 = TriRes(out_dc1)
        out_dc2 = self.dc2(out_dc1)
        out_dc2, res_dc2 = TriRes(out_dc2)
        out_dc3 = self.dc3(out_dc2)
        out_dc3, res_dc3 = TriRes(out_dc3)
        out_dc4 = self.dc4(out_dc3)
        out_dc4, res_dc4 = TriRes(out_dc4)
        out_dc5 = self.dc5(out_dc4)
        output = torch.cat((res_dc1,res_dc2,res_dc3,res_dc4,out_dc5),dim=1)
        output = self.conv1(output)
        return output

'''Frequency Division Residual Preservation'''
def TriRes(input, n_dis = 8, n_share = 8):
    _,C,_,_ = input.shape
    n_rem = int(C-n_dis-n_share)
    distillation, share, remain = torch.split(input, (n_dis, n_share, n_rem), dim=1)
    res_features = torch.cat((distillation,share),dim=1)
    learn_features = torch.cat((remain,share),dim=1)
    return  learn_features, res_features

'''Extcation-Distillation Residual'''
def EDR(input, n_dis = 8):
    _,C,_,_ = input.shape
    n_rem = int(C-n_dis)
    distillation, remain = torch.split(input, (n_dis, n_rem), dim=1)
    return  remain, distillation

'''Frequency Division Residual Groups'''
class FDRG(nn.Module):
    def __init__(self, nf=64, kernel_size=1, n_fd = 4, n_residual = 1, res_scale=1):
        super(FDRG, self).__init__()
        '''
        nf is the inpute channels
        kernel_size is the size of convolution kernel
        n_fd is the number of frequency division 
        n_reesidual is the number of Blocks
        res_scale is the residual coefficient 
        '''
        self.n_dc = n_fd
        self.nc_dc = int(nf//n_fd)
        self.res_scale = res_scale

        FDConv = [
            blocks.conv_layer(nf, nf, kernel_size),
            blocks.activation('prelu', in_channels = nf),
            blocks.conv_layer(nf, nf, kernel_size, groups=2)
        ]
        self.FDConv = nn.Sequential(*FDConv)

        FD_list = []
        for i in range(self.n_dc):          #
            FD_sublist = []
            for _ in range(n_residual):     #
                FD_sublist.append(blocks.ResBlock(self.nc_dc * (i + 1)))
            FD_list.append(nn.Sequential(*FD_sublist))
        self.FD_list = nn.Sequential(*FD_list)

    def forward(self, input):
        y = self.FDConv(input)
        FD_features = torch.split(y,self.nc_dc,dim=1)

        DC_out = locals()
        for i in range(self.n_dc):
            if i == 0 :
                DC_out['out_' + str(i)] = self.FD_list[i](FD_features[i])# + FD_features[i]
            else:
                DC_out['out_' + str(i)] = self.FD_list[i](DC_out['out_' + str(i-1)])# + DC_out['out_' + str(i-1)]
            if i != (self.n_dc -1) :
                DC_out['out_' + str(i)] = torch.cat((DC_out['out_' + str(i)],FD_features[i+1]),dim=1)

        output = DC_out['out_' + str(self.n_dc - 1)].mul(self.res_scale) + input
        return  output

