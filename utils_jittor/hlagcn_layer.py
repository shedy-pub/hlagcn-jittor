import math
import numpy as np
import jittor
import jittor.nn as nn
from jittor.nn import Module

class HLAGraphConvolution(Module):
    """LA-GCN module
    param:
          - node_num: total number of nodes
          - in_features: input feat dim
          - out_features: output feat dim
    input:
          - input[0]: feature map with size of (b, c, h, w)
          - input[1]: ratio information with size of (bs, 1)
    output:
          - x: refined feature maps with size of (b, c, h, w)
    """
    def __init__(self, node_num, in_features, out_features, bias=False, init='uniform', adj_op='combine'):
        super(HLAGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_op = adj_op
        if self.adj_op == 'content':
            self.trans_1 = nn.Linear(in_features, out_features, bias=bias)
            self.trans_2 = nn.Linear(in_features, out_features, bias=bias)
        elif self.adj_op == 'combine':
            self.trans_1 = nn.Linear(in_features, out_features, bias=bias)
            self.trans_2 = nn.Linear(in_features, out_features, bias=bias)
            self.x_square, self.y_square = generate_spatial_square(node_num)
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters(init)

    def reset_parameters(self, init):
        if init == 'uniform':
            stdv = 1. / math.sqrt(self.W.weight.size(1))
            nn.init.uniform_(self.W.weight, -stdv, stdv)
            nn.init.uniform_(self.trans_1.weight, -stdv, stdv)
            nn.init.uniform_(self.trans_2.weight, -stdv, stdv)
        elif init == 'xavier':
            nn.init.xavier_uniform_(self.W.weight)
            nn.init.xavier_uniform_(self.trans_1.weight)
            nn.init.xavier_uniform_(self.trans_2.weight)           

    def execute(self, input):
        [x, ratio_info] = input
        if self.adj_op == 'content':
            x1 = self.trans_1(x)
            x2 = self.trans_2(x)
            g_sim = jittor.matmul(x1, x2.permute(0, 2, 1))
            g_sim = nn.softmax(g_sim, dim=2)
            adj = g_sim
        elif self.adj_op == 'combine':
            x1 = self.trans_1(x)
            x2 = self.trans_2(x)
            g_sim = jittor.matmul(x1, x2.permute(0, 2, 1))
            g_sim = nn.softmax(g_sim, dim=2)
            g_spa = spatial_batch_matrix(self.x_square, self.y_square, ratio_info) 
            adj = (g_sim + g_spa) / 2
        h = jittor.matmul(adj, x)
        x = h + x
        x = self.W(x) 
        return x
    
class LAGCN1_Layer(nn.Module):
    """ the first LAGCN layer
    param:
          - node_num: original node num
          - in_features: input feat dim
          - out_features: output feat dim
          - num_gcn: gcn block number
    input:
          - input[0]: feature map with size of (b, c, h, w)
          - input[1]: ratio information with size of (bs, 1)
    output:
          - x: refined feature maps with size of (b, c, h, w)
    """
    
    def __init__(self, node_num, in_features, out_features, num_gcn, dropout=0.5, adj_op='combine'):
        super(LAGCN1_Layer, self).__init__()
        self.gcns = nn.ModuleList()
        self.num_gcn = num_gcn
        self.relu = nn.ReLU()
        for i in range(num_gcn):
            if i == 0:
                self.gcns.append(HLAGraphConvolution(node_num, in_features, out_features, adj_op=adj_op))
            else:
                self.gcns.append(HLAGraphConvolution(node_num, out_features, out_features, adj_op=adj_op))
    def execute(self, input):
        [x, ratio_info] = input
        bs, c, h, w = x.shape
        x = x.view(bs, c, -1).permute(0,2,1)
        for i in range(self.num_gcn):
            x = self.gcns[i]([x, ratio_info])
            if i != self.num_gcn-1:
                x = self.relu(x)
        x = x.permute(0,2,1).view(bs, c, h, w)        
        return x
    
class LAGCN2_Layer(nn.Module):
    """ the second LAGCN layer
    param:
          - node_num: aggregated node num
          - in_features: input feat dim
          - out_features: output feat dim
    input:
          - x: feature map with size of (b, c, h, w)
    output:
          - x: refined feature maps with size of (b, c, h, w)"""
    
    def __init__(self, node_num, in_features, out_features):
        super(LAGCN2_Layer, self).__init__()
        self.num_n = node_num
        self.num_s = out_features
        # reduce dim
        self.conv_state = nn.Conv2d(in_features, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = nn.Conv2d(in_features, self.num_n, kernel_size=1)
        # reasoning via graph convolution
        self.gcn = HLAGraphConvolution(self.num_n, self.num_s, self.num_s, adj_op='content')
        # extend dimension
        self.conv_extend = nn.Conv2d(self.num_s, in_features, kernel_size=1, bias=False)
        self.blocker = nn.BatchNorm2d(in_features, eps=1e-04) # should be zero initialized

    def execute(self, x):
        bs = x.size(0)
        x_state_reshaped = self.conv_state(x).view(bs, self.num_s, -1)
        x_proj_reshaped = self.conv_proj(x).view(bs, self.num_n, -1)
        x_rproj_reshaped = x_proj_reshaped
        x_n_state = jittor.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_rel = self.gcn([x_n_state.permute(0, 2, 1),[]])
        x_state_reshaped = jittor.matmul(x_n_rel.permute(0, 2, 1), x_rproj_reshaped)
        x_state = x_state_reshaped.view(bs, self.num_s, *x.size()[2:])
        x =  self.blocker(self.conv_extend(x_state))
        return x
    
def generate_spatial_square(N=49):
    num_node = int(math.sqrt(N))
    adj_cor_x, adj_cor_y = [], []
    for i in range(num_node):
        for j in range(num_node):
            adj_cor_x.append(i)
            adj_cor_y.append(j)
    adj_cor_x = np.array(adj_cor_x)[:,None]
    adj_cor_y = np.array(adj_cor_y)[:,None]
    adj_cor_x_t = np.transpose(adj_cor_x)
    adj_cor_y_t = np.transpose(adj_cor_y)
    x_square = np.square(np.abs(np.tile(adj_cor_x,(1,N))- np.tile(adj_cor_x_t,(N,1))))
    y_square = np.square(np.abs(np.tile(adj_cor_y,(1,N))- np.tile(adj_cor_y_t,(N,1))))
    x_square = jittor.float(x_square)
    x_square.requires_grad = False
    y_square = jittor.float(y_square)
    y_square.requires_grad = False
    return x_square, y_square

def spatial_batch_matrix(x_square, y_square, r=None, eps=1e-8, bs=10):
    N = x_square.shape[0]
    if r is not None:
        x_square=x_square
        y_square=y_square
        bs = r.shape[0]
        r_square = r.unsqueeze(1).pow(2)
        a = (x_square.unsqueeze(0) * r_square + y_square).sqrt() #(50,49,49)
    else:
        a = (x_square + y_square).sqrt() #(49,49)
        a = a.squeeze(0).repeat(bs, 1, 1)
    a_ = jittor.max(a, 1).unsqueeze(2)
    a_max = a_.repeat([1,1,N])
    a = a_max-a
    adj = a / ((a.view(bs*N, N).sum(dim=1).view(bs, N, -1)) + eps)
    return adj