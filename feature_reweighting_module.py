import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorchgo.utils import logger

class GNN_nl(nn.Module):
    def __init__(self, input_features, output_dim, gcn_layer, nf=96):#96
        super(GNN_nl, self).__init__()
        self.input_features = input_features
        self.nf = nf
        self.output_dim = output_dim #background and non-background

        self.num_layers = gcn_layer

        for i in range(self.num_layers):
            module_w = Wcompute(input_features=self.input_features + int(nf / 2) * i, nf = nf, operator = 'J2', activation = 'softmax', ratio = [2, 2, 1, 1])
            module_l = Gconv(nf_input = self.input_features + int(nf / 2) * i, nf_output = int(nf / 2), J = 2)
            self.add_module('layer_w{}'.format(i), module_w)
            self.add_module('layer_l{}'.format(i), module_l)

        self.w_comp_last = Wcompute(input_features=self.input_features + int(self.nf / 2) * self.num_layers, nf = nf, operator='J2', activation='softmax', ratio=[2, 2, 1, 1])
        self.layer_last = Gconv(nf_input = self.input_features + int(self.nf / 2) * self.num_layers, nf_output = self.output_dim, J = 2, bn_bool=False)

    def forward(self, x):#[40, 26, 133]
        W_init = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3))#[40, 26, 26, 1], the start of Graph Neural Network
        if True:#cuda==True:
            W_init = W_init.cuda()

        for i in range(self.num_layers):
            Wi = self._modules['layer_w{}'.format(i)](x, W_init)#calculate adjancent matrix

            #x_new = F.leaky_relu(self._modules['layer_l{}'.format(i)]([Wi, x])[1]),dongzhuoyao
            x_new = self._modules['layer_l{}'.format(i)]([Wi, x])#return [adjancent matrix, x_new]
            x_new = x_new[1]
            x_new = F.leaky_relu(x_new)

            x = torch.cat([x, x_new], 2)#residual intuition

        Wl=self.w_comp_last(x, W_init)
        out = self.layer_last([Wl, x])[1]

        return out[:, :, :]


def gmul(input):
    W, x = input
    # x is a tensor of size (bs, N, num_features). [40, 26, 181]
    # W is a tensor of size (bs, N, N, J), [40, 26, 26, 2]
    x_size = x.size()
    W_size = W.size()
    N = W_size[-2]
    W = W.split(1, 3)#split in dim=3, every part is 1
    W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)   x is a tensor of size (bs, N, num_features)
    output = torch.bmm(W, x) # output has size (bs, J*N, num_features), https://pytorch.org/docs/stable/torch.html#torch.bmm
    output = output.split(N, 1) #spli into  a list with length=J, size: [bs, N, num_features]
    output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
    return output


class Gconv(nn.Module):
    def __init__(self, nf_input, nf_output, J, bn_bool=False):
        super(Gconv, self).__init__()
        self.J = J
        self.num_inputs = J*nf_input
        self.num_outputs = nf_output
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

        self.bn_bool = bn_bool
        if self.bn_bool:
            self.bn = nn.BatchNorm1d(self.num_outputs)

    def forward(self, input):
        W = input[0]#input:[adjacent matrix, input feature]
        x = gmul(input) # out has size (bs, N, num_inputs)
        #if self.J == 1:
        #    x = torch.abs(x)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x = self.fc(x) # has size (bs*N, num_outputs)

        if self.bn_bool:
            x = self.bn(x)

        x = x.view(x_size[0],x_size[1], self.num_outputs)
        return W, x


class Wcompute(nn.Module):
    def __init__(self, input_features, nf, operator='J2', activation='softmax', ratio=[2,2,1,1], num_operators=1, drop=True, bn=False):
        super(Wcompute, self).__init__()
        logger.info("Wcompute activation type: {}".format(activation))
        self.num_features = nf
        self.operator = operator
        self.bn = bn
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        if bn:
            self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.drop = drop
        if self.drop:
            self.dropout = nn.Dropout(0.3)
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        if bn:
            self.bn_2 = nn.BatchNorm2d(int(nf * ratio[1]))
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), nf*ratio[2], 1, stride=1)
        if bn:
            self.bn_3 = nn.BatchNorm2d(nf*ratio[2])
        self.conv2d_4 = nn.Conv2d(nf*ratio[2], nf*ratio[3], 1, stride=1)
        if bn:
            self.bn_4 = nn.BatchNorm2d(nf*ratio[3])
        self.conv2d_last = nn.Conv2d(nf, num_operators, 1, stride=1)
        self.activation = activation

    def forward(self, x, W_id):
        #x : [40, 26, 133]
        #W_id:[40, 26, 26, 1]
        W1 = x.unsqueeze(2)
        W2 = torch.transpose(W1, 1, 2) #size: bs x N x N x num_features
        W_new = torch.abs(W1 - W2) #size: bs x N x N x num_features
        W_new = torch.transpose(W_new, 1, 3) #size: bs x num_features x N x N

        W_new = self.conv2d_1(W_new)
        if self.bn:
            W_new = self.bn_1(W_new)
        W_new = F.leaky_relu(W_new)
        if self.drop:
            W_new = self.dropout(W_new)

        W_new = self.conv2d_2(W_new)
        if self.bn:
            W_new = self.bn_2(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_3(W_new)
        if self.bn:
            W_new = self.bn_3(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_4(W_new)
        if self.bn:
            W_new = self.bn_4(W_new)
        W_new = F.leaky_relu(W_new)

        W_new = self.conv2d_last(W_new)
        W_new = torch.transpose(W_new, 1, 3) #size: bs x N x N x 1

        if self.activation == 'softmax':
            W_new = W_new - W_id.expand_as(W_new) * 1e8
            W_new = torch.transpose(W_new, 2, 3) # after transpose, [bs, N, 1, N]
            # Applying Softmax
            W_new = W_new.contiguous()
            W_new_size = W_new.size()
            W_new = W_new.view(-1, W_new.size(3))
            W_new = F.softmax(W_new)
            W_new = W_new.view(W_new_size)
            # Softmax applied
            W_new = torch.transpose(W_new, 2, 3)

        elif self.activation == 'sigmoid':
            W_new = F.sigmoid(W_new)
            W_new *= (1 - W_id)
        elif self.activation == 'none':
            W_new *= (1 - W_id)
        else:
            raise (NotImplementedError)




        if self.operator == 'laplace':
            W_new = W_id - W_new
        elif self.operator == 'J2':
            W_new = torch.cat([W_id, W_new], 3)
        else:
            raise(NotImplementedError)

        return W_new