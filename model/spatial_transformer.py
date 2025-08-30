import math
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()

        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                                stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_tcn_m(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=[1, 3, 7]):        # ks=9 initial
        super(unit_tcn_m, self).__init__()

        pad1 = int((kernel_size[0] - 1) / 2)
        pad2 = int((kernel_size[1] - 1) / 2)
        pad3 = int((kernel_size[2] - 1) / 2)

        mid_channels = out_channels//3

        self.conv11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.conv21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))
        self.conv31 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1))

        self.conv12 = nn.Conv2d(in_channels, mid_channels, kernel_size=(kernel_size[0], 1), padding=(pad1, 0),
                                stride=(stride, 1))
        self.conv22 = nn.Conv2d(in_channels, mid_channels, kernel_size=(kernel_size[1], 1), padding=(pad2, 0),
                                stride=(stride, 1))
        self.conv32 = nn.Conv2d(in_channels, mid_channels, kernel_size=(kernel_size[2], 1), padding=(pad3, 0),
                                stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv11)
        conv_init(self.conv21)
        conv_init(self.conv31)
        conv_init(self.conv12)
        conv_init(self.conv22)
        conv_init(self.conv32)
        bn_init(self.bn, 1)

    def forward(self, x):
        x1 = self.conv12(self.conv11(x))
        x2 = self.conv22(self.conv21(x))
        x3 = self.conv32(self.conv31(x))
        x = torch.cat([x1, x2, x3], dim=1)
        x  = self.bn(x)
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        # batchsize, channel, t, node_num
        N, C, T, V = x.size()
        #print(N, C, T, V)
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TCN_GCN_unit(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.tcn1 = unit_tcn_m(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        return self.relu(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hid_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hid_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.ReLU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hid_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.do2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        
        return x

class Spatial_Attention(nn.Module):
    def __init__(self, dim, out_dim, heads = 3, dropout = 0.1,kernel_size=1,stride=1,num_point =17,mask=None,group_mask=None):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.kernel_size = kernel_size
        self.num_point = num_point
        self.out_dim = out_dim
        self.scale = dim ** -0.5  # 1/sqrt(dim)
        self.kernel_size = 1
        self.padding = (self.kernel_size - 1) // 2
        self.stride = stride
        
        self.qkv_conv = nn.Conv2d(self.dim, 2 * self.dim + self.out_dim, kernel_size=self.kernel_size,
                                      stride=stride,
                                      padding=self.padding)
        
        self.attn_out = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1)
        self.key_rel = nn.Parameter(torch.randn(((self.num_point ** 2) - self.num_point, self.dim // self.heads), requires_grad=True))
        self.key_rel_diagonal = nn.Parameter(torch.randn((1, self.dim // self.heads), requires_grad=True))

        self.nn1 = nn.Linear(self.dim, self.out_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)        
        self.do1 = nn.Dropout(dropout)

        # Fix: Handle None mask case
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                # Initialize learnable matrix with ones having the same shape as mask
                self.PA = nn.Parameter(torch.ones_like(mask).float())
            else:
                # Initialize learnable matrix with ones having the same shape as mask
                mask_tensor = torch.from_numpy(mask.astype(np.float32))
                self.PA = nn.Parameter(torch.ones_like(mask_tensor).float())
        else:
            # Initialize with ones matrix if no mask provided
            self.PA = nn.Parameter(torch.ones(self.num_point, self.num_point).float())

    def forward(self, x, mask = None,group_mask=None):
        
        B, V, C = x.size() #B*T,V,C

        xa = x.permute(0, 2, 1).unsqueeze(2)  # from (B*T, V, C) → (B*T, C, 1, V)

        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(xa, self.dim, self.dim, self.heads)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        #logits *= self.scale  # scale the logits

        if mask is not None:
            logits = logits.reshape(-1, V, V)
            #print(logits.shape)
            M, V, V = logits.shape
            #print(mask.shape)
            #print(self.PA.shape)
            A = mask * self.PA  # Changed from in-place *= to regular multiplication
            #print(A.size())
            A = A.unsqueeze(0).expand(M, V, V)
            #print(A.size())
            #print("######")
            logits = logits + A
            logits = logits.reshape(B, self.heads, V, V)

        # Add group mask, if provided
        if group_mask is not None:
            #print(group_mask.shape)
            assert group_mask.shape == logits.shape, f"group_mask shape mismatch: {group_mask.shape} vs {logits.shape}"
            #logits = (logits+ group_mask) * 0.5
            logits = logits+ group_mask

        rel_logits = self.relative_logits(q)
        device = x.device

        logits_sum = torch.add(logits, rel_logits)
        weights = F.softmax(logits_sum, dim=-1)

        mask = torch.bernoulli((0.5) * torch.ones(B * self.heads * V, device=device))
        mask = mask.reshape(B, self.heads, V).unsqueeze(2).expand(B, self.heads, V, V)
        weights = weights * mask
        weights = weights / (weights.sum(3, keepdim=True) + 1e-8)

        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (B, self.heads, 1, V, self.dim // self.heads))
        attn_out = attn_out.permute(0, 1, 4, 2, 3)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        attn_out = attn_out.permute(0, 2, 3, 1).contiguous()  # (B, T, V, C) T=1 in this case
        
        out = attn_out.squeeze(1).view(-1, attn_out.size(2), attn_out.size(3)) # (B*T, V, C)
        
        out =  self.nn1(out)
        out = self.do1(out)
        
        return out

    def compute_flat_qkv(self, x, dk, dv, Nh):      
        qkv = self.qkv_conv(x)
        # T=1 in this case, because we are considering each frame separately
        N, _, T, V = qkv.size()

        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q = q*(dkh ** -0.5)
            
        flat_q = torch.reshape(q, (N, Nh, dkh, T * V))
        flat_k = torch.reshape(k, (N, Nh, dkh, T * V))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, T * V))
            
        return flat_q, flat_k, flat_v, q, k, v   
        
    def split_heads_2d(self, x, Nh):
        B, channels, T, V = x.size()
        ret_shape = (B, Nh, channels // Nh, T, V)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, T, V = x.size()
        ret_shape = (batch, Nh * dv, T, V)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, T, V = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)
        q_first = q.unsqueeze(4).expand((B, Nh, T, V, V - 1, dk))
        q_first = torch.reshape(q_first, (B * Nh * T, -1, dk))

        # q used to multiply for the embedding of the parameter on the diagonal
        q = torch.reshape(q, (B * Nh * T, V, dk))
        # key_rel_diagonal: (1, dk) -> (V, dk)
        param_diagonal = self.key_rel_diagonal.expand((V, dk))
        rel_logits = self.relative_logits_1d(q_first, q, self.key_rel, param_diagonal, T, V, Nh)
        return rel_logits

    def relative_logits_1d(self, q_first, q, rel_k, param_diagonal, T, V, Nh):
        # compute relative logits along one dimension
        # (B*Nh*1,V^2-V, self.dk // Nh)*(V^2 - V, self.dk // Nh)

        # (B*Nh*1, V^2-V)
        rel_logits = torch.einsum('bmd,md->bm', q_first, rel_k)
        # (B*Nh*1, V)
        rel_logits_diagonal = torch.einsum('bmd,md->bm', q, param_diagonal)

        # reshapes to obtain Srel
        rel_logits = self.rel_to_abs(rel_logits, rel_logits_diagonal)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, V, V))
        return rel_logits

    def rel_to_abs(self, rel_logits, rel_logits_diagonal):
        B, L = rel_logits.size()
        B, V = rel_logits_diagonal.size()

        # (B, V-1, V) -> (B, V, V)
        rel_logits = torch.reshape(rel_logits, (B, V - 1, V))
        row_pad = torch.zeros(B, 1, V).to(rel_logits)
        rel_logits = torch.cat((rel_logits, row_pad), dim=1)

        # concat the other embedding on the left
        # (B, V, V) -> (B, V, V+1) -> (B, V+1, V)
        rel_logits_diagonal = torch.reshape(rel_logits_diagonal, (B, V, 1))
        rel_logits = torch.cat((rel_logits_diagonal, rel_logits), dim=2)
        rel_logits = torch.reshape(rel_logits, (B, V + 1, V))

        # slice
        flat_sliced = rel_logits[:, :V, :]
        final_x = torch.reshape(flat_sliced, (B, V, V))
        return final_x


class Spatial_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_point=17, mask=None,group_mask=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        if dim == mlp_dim:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Residual(Spatial_Attention(dim, mlp_dim, heads = heads, dropout = dropout, num_point=num_point, mask=mask,group_mask=group_mask)),
                    Residual(LayerNormalize(mlp_dim, MLP_Block(mlp_dim, mlp_dim*2, dropout = dropout)))
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Spatial_Attention(dim, mlp_dim, heads = heads, dropout = dropout, num_point=num_point, mask=mask,group_mask=group_mask),
                    Residual(LayerNormalize(mlp_dim, MLP_Block(mlp_dim, mlp_dim*2, dropout = dropout)))
                ]))
    def forward(self, x, mask = None,group_mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask = mask,group_mask=group_mask) # go to attention
            x = mlp(x) #go to MLP_Block
        return x


class TCN_STRANSF_unit(nn.Module):
    def __init__(self, in_channels, out_channels,num_point, heads=3, stride=1, residual=True, dropout=0.1, mask=None, mask_grad=True, group_mask=None):
        super(TCN_STRANSF_unit, self).__init__()
        # Pass mask to Spatial_Transformer
        self.transf1 = Spatial_Transformer(dim=in_channels, depth=3, heads=heads, mlp_dim=in_channels, dropout=dropout, num_point=num_point, mask=mask,group_mask=group_mask)
        self.tcn1 = unit_tcn_m(in_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)
        
        if mask is not None:
            self.mask = nn.Parameter(mask, requires_grad=mask_grad)
        else:
            self.mask = None

    def forward(self, x, mask=None,group_mask=None):
        B, C, T, V= x.size()  
        
        tx = x.permute(0, 2, 3, 1).contiguous().view(B * T, V, C)
        if mask==None:
            tx = self.transf1(tx, self.mask,group_mask=group_mask) #change input to x
        else:
            tx = self.transf1(tx, mask,group_mask=group_mask) #change input to x
        tx = tx.view(B, T, V, C).permute(0, 3, 1, 2).contiguous()

        x = self.tcn1(tx) + self.residual(x)
        return self.relu(x)


class PoseFormer(nn.Module):
    def __init__(self, in_channels=3, num_person=12, num_point=17, num_head=6, graph=None, graph_args=dict()):
        super(ZiT, self).__init__()
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        bn_init(self.data_bn, 1)
        self.heads = num_head

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        #self.A = A[0] + A[1] + A[2]

        #self.A = torch.from_numpy(self.graph.A[0].astype(np.float32))
        #self.A = torch.from_numpy((self.graph.A[0] + self.graph.A[1] + self.graph.A[2]).astype(np.float32))
        self.A = torch.from_numpy((self.graph.A[0]).astype(np.float32))
        self.l1 = TCN_GCN_unit(3, 48, self.graph.A, residual=False)
        self.l2 = TCN_STRANSF_unit(48, 48, num_point=num_point, heads=num_head, mask=self.A, mask_grad=False)      
        self.l3 = TCN_STRANSF_unit(48, 48, num_point=num_point,heads=num_head, mask=self.A, mask_grad=False)
        self.l4 = TCN_STRANSF_unit(48, 96, num_point=num_point,heads=num_head, stride=2, mask=self.A, mask_grad=True)
        self.l5 = TCN_STRANSF_unit(96, 96, num_point=num_point,heads=num_head, mask=self.A, mask_grad=True)
        self.l6 = TCN_STRANSF_unit(96, 192, num_point=num_point, heads=num_head, stride=2, mask=self.A, mask_grad=True)
        self.l6 = TCN_STRANSF_unit(96, 192, num_point=num_point, heads=num_head, stride=2, mask=self.A, mask_grad=True)
        self.l7 = TCN_STRANSF_unit(192, 192, num_point=num_point ,heads=num_head, mask=self.A, mask_grad=True)


    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)

        B, C_, T_, V_ = x.size()
        x = x.view(N, M, C_, T_, V_).mean(4)
        x = x.permute(0, 2, 3, 1).contiguous()

        return x


class SceneFormer(nn.Module):
    def __init__(self, num_class=8, num_head=6,num_person=2):
        super(ZoT, self).__init__()

        self.heads = num_head

        self.conv1 = nn.Conv2d(192, num_head,kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(192, num_head,kernel_size=(1, 1))
        conv_init(self.conv1)
        conv_init(self.conv2)

        self.l1 = TCN_STRANSF_unit(192, 276, num_point=num_person, heads=num_head,mask=None)       # 192 276
        self.l2 = TCN_STRANSF_unit(276, 276,num_point=num_person,  heads=num_head,mask =None)

        self.fc = nn.Linear(276, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))

        self.group_mask = None
    
    def forward(self, x):
        # N,C,T,M
        x1 = self.conv1(x)  # [N, heads, T, M]
        x2 = self.conv2(x)  # [N, heads, T, M]

        # Create pairwise difference
        x1 = x1.unsqueeze(4)  # [N, heads, T, M, 1]
        x2 = x2.unsqueeze(3)  # [N, heads, T, 1, M]
        mask = x1 - x2        # [N, heads, T, M, M]

        # Softmax along the last dim to make it attention-like
        group_mask = mask.softmax(dim=-1)  # still [N, heads, T, M, M]

        # Match expected shape: [B, heads, V, V] = [N, heads, M, M]
        # So merge batch and time dims
        N, heads, T, M, _ = group_mask.shape
        group_mask = group_mask.permute(0, 2, 1, 3, 4).contiguous().view(N * T, heads, M, M).detach()

        #print(group_mask.shape)
        self.group_mask = group_mask
        #print(group_mask.shape)
        x = self.l1(x, group_mask=group_mask)
        x = self.l2(x,group_mask=group_mask )
        x = x.mean(3).mean(2)

        return self.fc(x)

class Model(nn.Module):
    def __init__(self, num_class=8, in_channels=3, num_person=12, num_point=17, num_head=6, graph=None, graph_args=dict()):
        super(Model, self).__init__()

        self.body_transf = PoseFormer(in_channels=in_channels, num_person=num_person, num_point=num_point, num_head=num_head, graph=graph, graph_args=graph_args)
        self.group_transf = SceneFormer(num_class=num_class, num_head=num_head, num_person=num_person)


    def forward(self, x):
        x = self.body_transf(x)
        x = self.group_transf(x)

        return x
    