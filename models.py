import os
os.environ['DGLBACKEND'] = 'pytorch'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import uuid
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.checkpoint import checkpoint

import dgl

FOV_NET_DICT = {
    "unet_3_pad0": {
        3: {
            "fov": 67,
            "jump": 8,
        },
        4: {
            "fov": 79,
            "jump": 4,
        },
        5: {
            "fov": 85,
            "jump": 2,
        },
        6: {
            "fov": 88,
            "jump": 1,
        }
    }
}


def l2norm(inp, dim):
    return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

def set_trainable(model, trainable):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable

def print_parameter(model):
    child_counter = 0
    for name, parameter in model.named_parameters():
        print("parameter_name:{}, grad_flag:{}".format(name, parameter.requires_grad))
        child_counter += 1

class TSUnet(nn.Module):

    def __init__(self, net1_param, net2_param, ds_rate, norm_method='bn', act_method='relu',
                 return_res=False, training_stage=0):
        super(TSUnet, self).__init__()
        self.unet1 = Unet3DMultitaskGeneric(**net1_param) if net1_param is not None else None
        self.unet2 = Unet3DMultitaskGeneric(**net2_param) if net1_param is not None else None
        if net1_param is not None and net2_param is not None:
            net1_param['norm_method'] = norm_method
            net2_param['norm_method'] = norm_method
            net1_param['act_method'] = act_method
            net2_param['act_method'] = act_method
            print(net1_param)
            print(net2_param)
            print("INPL!")
        # self.layer_norm = nn.GroupNorm(1, net2_param['in_ch_list'][0])
        self.sanity_check = False
        self.ds_rate = ds_rate
        # self.ds_rate = 0.5
        self.return_res = return_res
        self.training_stage = training_stage
        self.inference_cache = {}

    def set_trace_path(self, *args):
        self.unet1.set_trace_path(*args) if hasattr(self.unet1, "set_trace_path") else None
        self.unet2.set_trace_path(*args) if hasattr(self.unet2, "set_trace_path") else None

    def set_train_stage(self):
        # 0 stage 1 only
        # 1 stage 2 only
        # 2 stage 1 and 2
        if self.training_stage == 0:
            set_trainable(self.unet2, False)
            set_trainable(self.unet1, True)
            self.unet1.checkpoint_layers[0] = 1
            self.unet2.checkpoint_layers[0] = 1
        elif self.training_stage == 1:
            set_trainable(self.unet1, False)
            set_trainable(self.unet2, True)
            self.unet1.checkpoint_layers[0] = 0
            self.unet2.checkpoint_layers[0] = 0
        elif self.training_stage == 3:
            self.unet1.finetune()
            self.unet2.finetune()
        else:
            set_trainable(self.unet1, True)
            set_trainable(self.unet2, True)
            self.unet1.checkpoint_layers[0] = 1
            self.unet2.checkpoint_layers[0] = 1
        print_parameter(self)

    def init(self, initializer):
        initializer.initialize(self)
        self.unet1.init(initializer)
        self.unet2.init(initializer)

    def scan_level_inference(self, scan):
        ds_scan = F.interpolate(scan.float(), scale_factor=self.ds_rate, mode='trilinear', align_corners=True, recompute_scale_factor=True)
        if torch.cuda.is_available() and self.is_cuda:
            ds_scan = ds_scan.cuda()
        out11 = self.unet1(ds_scan)[0].cpu().float()
        # out11 = self.unet1(ds_scan)[0]
        r_out11 = F.interpolate(out11, size=scan.shape[-3:], mode='trilinear', align_corners=True, recompute_scale_factor=False)
        inf11 = F.softmax(r_out11, dim=1)
        return inf11

    def forward(self, pad_scan, chunks, slices_list):
        assert (pad_scan.dim() == 5)
        assert (chunks.dim() == 5)

        # step.1 downsample pad_scan
        if pad_scan.type() == "torch.HalfTensor":
            previous_t = pad_scan.type()
            pad_scan = pad_scan.float()
        else:
            previous_t = None
        ds_pad_scan = F.interpolate(pad_scan, scale_factor=self.ds_rate, mode='trilinear', align_corners=True, recompute_scale_factor=True)
        if previous_t is not None:
            ds_pad_scan = ds_pad_scan.type(previous_t)
        if torch.cuda.is_available() and self.is_cuda:
            ds_pad_scan = ds_pad_scan.cuda()
        out11, out12 = self.unet1(ds_pad_scan)
        if out11.type() == "torch.HalfTensor":
            out11 = out11.float()
        if out12.type() == "torch.HalfTensor":
            out11 = out11.float()
        # out11 lobes, out12, lobe borders
        assert (out11.shape[-3:] == ds_pad_scan.shape[-3:])
        assert (out12.shape[-3:] == ds_pad_scan.shape[-3:])
        # step.2 reconstruct inference from stage 1
        r_out11 = F.interpolate(out11, size=pad_scan.shape[-3:], mode='trilinear', align_corners=True, recompute_scale_factor=False)
        r_out12 = F.interpolate(out12, size=pad_scan.shape[-3:], mode='trilinear', align_corners=True, recompute_scale_factor=False)
        inf11 = F.softmax(r_out11, dim=1)
        inf12 = torch.sigmoid(r_out12)
        # print("inf11.shape[-3:] :{}, pad_scan.shape[-3:]:{} ".format(inf11.shape[-3:], pad_scan.shape[-3:]))
        assert (inf11.shape[-3:] == pad_scan.shape[-3:])
        assert (inf12.shape[-3:] == pad_scan.shape[-3:])
        # pad since it is valid padding mode at second stage.
        pad_size = [(((ss.stop - ss.start) - (cs.stop - cs.start)) // 2,
                     ((ss.stop - ss.start) - (cs.stop - cs.start)) - ((ss.stop - ss.start) - (cs.stop - cs.start)) // 2)
                    for ss, cs in zip(slices_list[0][0], slices_list[0][-1])]
        # depad_slices = tuple([slice(ps[0], -ps[1]) if ps[1] != 0 else slice(ps[0], None) for ps in pad_size])
        pad_inf11 = F.pad(inf11, tuple(np.asarray(pad_size[::-1]).flatten().tolist()), "constant", 0)
        pad_inf12 = F.pad(inf12, tuple(np.asarray(pad_size[::-1]).flatten().tolist()), "constant", 0)
        # now we can slice out the stage 1 output according to stage 2's position.

        stage_2_input_list = []
        out11_crop_list = []
        out12_crop_list = []
        crop_slice_list = []
        for slices, chunk in zip(slices_list, chunks):
            crop_slice = (slice(None, None), slice(None, None)) + slices[0]
            crop_output_slice = (slice(None, None), slice(None, None)) + slices[-1]
            crop_slice_list.append(crop_slice)
            pad_crop_inf11 = pad_inf11[crop_slice]
            pad_crop_inf12 = pad_inf12[crop_slice]
            if self.return_res:
                crop_r_out11 = r_out11[crop_output_slice]
                out11_crop_list.append(crop_r_out11)
                out12_crop_list.append(r_out12[crop_output_slice])
            stage2_input = torch.cat([chunk.unsqueeze(0), pad_crop_inf11, pad_crop_inf12], dim=1)
            stage_2_input_list.append(stage2_input)
        out21, out22 = self.unet2(torch.cat(stage_2_input_list, dim=0),
                                  (crop_slice_list, pad_inf11.shape[-3:]))
        if self.return_res:
            return out11, out12, out21 + torch.cat(out11_crop_list, dim=0), \
                   out22 + torch.cat(out12_crop_list, dim=0)
        return out11, out12, out21, out22

    def inference(self, pad_scan, chunks, slices_list):
        pad_size = [(((ss.stop - ss.start) - (cs.stop - cs.start)) // 2,
                     ((ss.stop - ss.start) - (cs.stop - cs.start)) - ((ss.stop - ss.start) - (cs.stop - cs.start)) // 2)
                    for ss, cs in zip(slices_list[0][0], slices_list[0][-1])]
        # depad_slices = tuple([slice(ps[0], -ps[1]) if ps[1] != 0 else slice(ps[0], None) for ps in pad_size])

        if len(self.inference_cache.keys()) == 0:
            assert (pad_scan.dim() == 5)
            assert (chunks.dim() == 5)
            if pad_scan.type() == "torch.HalfTensor":
                previous_t = pad_scan.type()
                pad_scan = pad_scan.float()
            else:
                previous_t = None
            ds_pad_scan = F.interpolate(pad_scan, scale_factor=self.ds_rate, mode='trilinear', align_corners=True, recompute_scale_factor=True)
            if torch.cuda.is_available() and self.is_cuda:
                ds_pad_scan = ds_pad_scan.cuda()
            if previous_t is not None:
                ds_pad_scan = ds_pad_scan.type(previous_t)
            out11, out12 = self.unet1(ds_pad_scan)
            assert (out11.shape[-3:] == ds_pad_scan.shape[-3:])
            assert (out12.shape[-3:] == ds_pad_scan.shape[-3:])
            if out11.type() == "torch.HalfTensor":
                out11 = out11.float()
            if out12.type() == "torch.HalfTensor":
                out12 = out12.float()
            # step.2 reconstruct inference from stage 1
            r_out11 = F.interpolate(out11, size=pad_scan.shape[-3:], mode='trilinear', align_corners=True, recompute_scale_factor=False)
            r_out12 = F.interpolate(out12, size=pad_scan.shape[-3:], mode='trilinear', align_corners=True, recompute_scale_factor=False)
            inf11 = F.softmax(r_out11, dim=1)
            inf12 = torch.sigmoid(r_out12)
            # print("inf11.shape[-3:] :{}, pad_scan.shape[-3:]:{} ".format(inf11.shape[-3:], pad_scan.shape[-3:]))
            assert (inf11.shape[-3:] == pad_scan.shape[-3:])
            assert (inf12.shape[-3:] == pad_scan.shape[-3:])
            # pad since it is valid padding mode at second stage.
            pad_inf11 = F.pad(inf11, tuple(np.asarray(pad_size[::-1]).flatten().tolist()), "constant", 0)
            pad_inf12 = F.pad(inf12, tuple(np.asarray(pad_size[::-1]).flatten().tolist()), "constant", 0)

            self.inference_cache = {
                "pad_inf11": pad_inf11,
                "pad_inf12": pad_inf12,
                "r_out11": r_out11,
            }
        else:
            pad_inf11 = self.inference_cache['pad_inf11']
            pad_inf12 = self.inference_cache['pad_inf12']
            r_out11 = self.inference_cache['r_out11']

        stage_2_input_list = []
        out11_crop_list = []
        crop_slice_list = []
        for slices, chunk in zip(slices_list, chunks):
            crop_slice = (slice(None, None), slice(None, None)) + slices[0]
            crop_output_slice = (slice(None, None), slice(None, None)) + slices[-1]
            crop_slice_list.append(crop_slice)
            pad_crop_inf11 = pad_inf11[crop_slice]
            pad_crop_inf12 = pad_inf12[crop_slice]
            if self.return_res:
                crop_r_out11 = r_out11[crop_output_slice]
                out11_crop_list.append(crop_r_out11)

            stage2_input = torch.cat([chunk.unsqueeze(0), pad_crop_inf11, pad_crop_inf12], dim=1)
            stage_2_input_list.append(stage2_input)
        out21, out22 = self.unet2(torch.cat(stage_2_input_list, dim=0), (crop_slice_list, pad_inf11.shape[-3:]))
        if self.return_res:
            return F.softmax(out21 + torch.cat(out11_crop_list, dim=0), dim=1)
        return F.softmax(out21, dim=1)


def non_local_merge(f, method):
    if method == "softmax":
        return F.softmax(f, dim=-1)
    elif method == 'dot':
        return f / f.shape[-1]
    else:
        raise NotImplementedError


def geo_transform(a, b, method):
    if method == 'rbf':
        ea = torch.exp(a)
        eb = torch.exp(b)
        return torch.matmul(ea, ea.transpose(-1, -2)) - \
               2 * torch.matmul(ea, eb) \
               + torch.matmul(eb.transpose(-1, -2), eb)
    elif method == 'dot':
        return torch.matmul(a, b)
    else:
        raise NotImplementedError



def non_local_metric_wrapper(non_local_metric_method):
    if non_local_metric_method == "gaussian":
        return non_local_gaussian
    elif non_local_metric_method == "scale_dot_product":
        return non_local_scale_dot_product


def non_local_gaussian(x_theta, x_phi):
    x_phi = x_phi.view(*x_phi.shape[:-3], -1)
    x_theta = x_theta.view(*x_theta.shape[:-3], -1)
    x_theta = x_theta.permute(0, 2, 1).contiguous()
    f = torch.matmul(x_theta, x_phi)
    f_sm = F.softmax(f, dim=-1)
    return f_sm


def non_local_scale_dot_product(x_theta, x_phi):
    x_phi = x_phi.view(*x_phi.shape[:-3], np.prod(x_phi.shape[-3:]))
    x_theta = x_theta.transpose(1, -1).contiguous()
    x_theta = x_theta.view(x_theta.shape[0], np.prod(x_theta.shape[1:-1]), x_theta.shape[-1])
    f = torch.matmul(x_theta, x_phi)
    N = f.size(-1)
    f_div_C = f / N
    return f_div_C

def normal_wrapper(normal_method, in_ch, in_ch_div=2):
    if normal_method == "bn":
        return nn.BatchNorm3d(in_ch)
    elif normal_method == "bnt":
        # this should be used when batch_size=1
        return nn.BatchNorm3d(in_ch, affine=True, track_running_stats=False)
    elif normal_method == "bntna":
        # this should be used when batch_size=1
        return nn.BatchNorm3d(in_ch, affine=False, track_running_stats=False)
    elif normal_method == "ln":
        return nn.GroupNorm(1, in_ch)
    elif normal_method == "lnna":
        return nn.GroupNorm(1, in_ch, affine=False)
    elif normal_method == "in":
        return nn.GroupNorm(in_ch, in_ch)
    else:
        return nn.GroupNorm(in_ch_div, in_ch)


def act_wrapper(act_method, num_parameters=1, init=0.25):
    if act_method == "relu":
        return nn.ReLU(inplace=True)
    elif act_method == "prelu":
        return nn.PReLU(num_parameters, init)
    else:
        raise NotImplementedError


def checkpoint_wrapper(module, segments, *tensors):
    if segments > 0:
        # if type(module) in [nn.Sequential, nn.ModuleList, list]:
        #     return checkpoint_sequential(module, segments, *tensors)
        # else:
        return checkpoint(module, *tensors)
    else:
        return module(*tensors)


def crop_concat_5d(t1, t2):
    """"Channel-wise cropping for 5-d tensors in NCDHW format,
    assuming t1 is smaller than t2 in all DHW dimension. """
    assert (t1.dim() == t2.dim() == 5)
    assert (t1.shape[-1] <= t2.shape[-1])
    slices = (slice(None, None), slice(None, None)) \
             + tuple(
        [slice(int(np.ceil((b - a) / 2)), a + int(np.ceil((b - a) / 2))) for a, b in zip(t1.shape[2:], t2.shape[2:])])
    x = torch.cat([t1, t2[slices]], dim=1)
    return x

class CNet(nn.Module):

    def __init__(self, in_ch, g_dim, d_sim, pool_strides=[1, 1],
                 drop_rate=0.0, merge_method='softmax', group_norm=True):
        super(CNet, self).__init__()
        self.d_sim = d_sim
        self.g_dim = g_dim
        self.in_ch = in_ch
        self.pool_strides = pool_strides
        self.drop_rate = drop_rate
        self.merge_method = merge_method
        self.G = nn.Linear(in_ch, g_dim)
        self.theta = nn.Linear(in_ch, d_sim)
        self.phi = nn.Linear(in_ch, d_sim)
        self.bn = nn.BatchNorm3d(in_ch)
        if self.drop_rate > 0:
            print("CrossNet using dropout: {}".format(self.drop_rate))
            self.r = nn.Sequential(
                nn.Linear(g_dim, in_ch),
                nn.Dropout(self.drop_rate)
            )
        else:
            self.r = nn.Sequential(
                nn.Linear(g_dim, in_ch),
            )
        self.plot_path = None
        self.bns = nn.ModuleList(
            [nn.GroupNorm(in_ch // 4, in_ch) for _ in self.pool_strides]
        )
        self.group_norm = group_norm
        self.cn_num = 0

    def finetune(self):
        pass

    def init(self, initializer):
        initializer.initialize(self)
        nn.init.constant(self.G.weight, 0.0)
        nn.init.constant(self.G.bias, 0.0)

        nn.init.constant(self.r[0].weight, 0.0)
        nn.init.constant(self.r[0].bias, 0.0)
        nn.init.constant(self.theta.weight, 0.0)
        nn.init.constant(self.theta.bias, 0.0)
        nn.init.constant(self.phi.weight, 0.0)
        nn.init.constant(self.phi.bias, 0.0)

    def init_graph(self, spatial_size, pool_stride, fv=None):
        n_nodes_total = int(np.prod(spatial_size))
        g = nx.empty_graph(n_nodes_total, create_using=nx.DiGraph)
        for n in range(n_nodes_total):
            s_n = list(np.unravel_index(n, spatial_size))
            cross_n = []
            for d in range(len(spatial_size)):
                for a in list(range(0, spatial_size[d], pool_stride)):
                    temp = s_n[:]
                    temp[d] = a
                    cross_n.append(temp)
            if s_n not in cross_n:
                cross_n.append(s_n)
            end_nodes = np.ravel_multi_index(np.asarray(cross_n).T, spatial_size)
            end_nodes = list(set(end_nodes))
            g.add_edges_from(list(zip(end_nodes, [n] * len(end_nodes))))

        graph = dgl.from_networkx(g)
        return graph

    def compute_cross_x(self, e_xs, n_x):
        # appearance term
        # e_xs: node_batchs, edge_batches, batches, channels
        # n_x: node_batchs, batches, channels
        # e_ps: node_batchs, edge_batches, batches, 3
        # n_p: node_batchs, batches, 3
        node_batches = e_xs.shape[0]
        edge_batches = e_xs.shape[1]
        batches = e_xs.shape[2]
        in_ch = e_xs.shape[3]
        x_phi = self.phi(e_xs.view(-1, in_ch)).reshape(node_batches, edge_batches, batches, self.d_sim)
        x_theta = self.theta(n_x.view(-1, in_ch)).reshape(node_batches, batches, self.d_sim).unsqueeze(1)
        x_theta = x_theta.permute(2, 0, 1, 3).contiguous()  # [batches, node_batches, 1, d_sim]
        x_phi = x_phi.permute(2, 0, 3, 1).contiguous()  # [batches, node_batches, d_sim, edge_batches]
        f = torch.matmul(x_theta, x_phi)  # [batches, node_batches, 1, edge_batches]
        f_sm = non_local_merge(f, self.merge_method)
        # if self.cn_num % 100 == 0:
        #     print("FM:{}, FV:{}.".format(f.mean().item(), f.var().item()))
        x_g = self.G(e_xs.view(-1, self.in_ch)).reshape(node_batches, edge_batches, batches, self.g_dim)
        x_g = x_g.permute(2, 0, 1, 3).contiguous()  # [batches, node_batches, edge_batches, g_dim]
        y = torch.matmul(f_sm, x_g)  # x_g is [batches, node_batches, 1, self.g_dim]
        cross_x = self.r(y.view(-1, self.g_dim)).reshape(batches, node_batches, in_ch)
        cross_x = cross_x.permute(1, 0, 2).contiguous()  # [node_batches, batches, in_ch]
        return cross_x

    def plot_attention_map(self, x, crop_slice_list, scan_shape, n_layers, padding, plot_key):
        pass


    def message_func(self, edges):
        e_xs = edges.src['x']
        return {'e_xs': e_xs}

    def reduce_func(self, nodes):
        e_xs = nodes.mailbox['e_xs']
        n_x = nodes.data['x']
        cross_x = self.compute_cross_x(e_xs, n_x)
        return {'cross_x': cross_x}

    def forward(self, x, args=None):
        if args is not None:
            crop_slice_list = args[0]
            scan_shape = args[1]
            n_layers = args[2]
            padding = args[3]
        else:
            crop_slice_list = None
            scan_shape = None
            n_layers = None
            padding = None
        spatial_sizes = x.shape[-3:]
        h = x
        uid = uuid.uuid4()
        if self.plot_path is not None:
            self.plot_attention_map(x, crop_slice_list, scan_shape, n_layers, padding, "{}".format(uid))
        g = self.init_graph(spatial_sizes, 1, fv=h)
        g.to(x.device)
        for i, pool_stride in enumerate(self.pool_strides):
            h_flat = h.view(*h.shape[:-3], -1)
            h_flat = h_flat.permute(2, 0, 1).contiguous()
            g.ndata['x'] = h_flat
            g.update_all(self.message_func, self.reduce_func)
            cross_x = g.ndata.pop('cross_x')
            cross_x = cross_x.permute(1, 2, 0).contiguous()
            cross_x = cross_x.view(x.shape)
            if self.plot_path is not None:
                self.plot_gradients(cross_x, x, "{}_{}".format(uid, i))
            if self.group_norm:
                h = h + self.bns[i](cross_x)
            else:
                h = h + cross_x
            g.ndata.pop('x')
        g.clear()
        self.cn_num += 1
        return F.relu(self.bn(h), inplace=True)


class CBNet(nn.Module):

    def __init__(self, in_ch, g_dim, d_sim, p_sim, pool_strides=[1, 1, 1],
                 drop_rate=0.0, merge_method="softmax", group_norm=False, do_norm=True):
        super(CBNet, self).__init__()
        self.d_sim = d_sim
        self.g_dim = g_dim
        self.p_sim = p_sim
        self.in_ch = in_ch
        self.group_norm = group_norm
        self.pool_strides = pool_strides
        self.merge_method = merge_method
        self.drop_rate = drop_rate
        self.G = nn.Linear(in_ch, g_dim)
        self.theta = nn.Linear(in_ch, d_sim)
        self.phi = nn.Linear(in_ch, d_sim)
        if self.drop_rate > 0:
            self.r = nn.Sequential(
                nn.Linear(g_dim, in_ch),
                nn.Dropout(self.drop_rate)
            )
        else:
            self.r = nn.Sequential(
                nn.Linear(g_dim, in_ch),
            )
        self.geo_theta = nn.Linear(3, self.p_sim)
        self.geo_phi = nn.Linear(3, self.p_sim)
        self.bns = nn.ModuleList(
            [nn.GroupNorm(in_ch // 4, in_ch) for _ in self.pool_strides]
        )
        self.plot_path = None
        self.cn_num = 0
        self.do_norm = do_norm
        self.bn = nn.BatchNorm3d(in_ch)

    def finetune(self):
        set_trainable(self, True)

    def build_geo_feature(self, x, crop_slice_list, scan_shape, n_layers, padding, normalize=True):
        spatial_size = x.shape[-3:]
        t = torch.ones(spatial_size).type(x.type())
        p = torch.nonzero(t).type(x.type())
        if crop_slice_list is None:
            if normalize:
                p /= (torch.Tensor(list(spatial_size)).type(x.type()) - 1.0)
                p -= 0.5
                # p -= p.mean()
                # p /= p.std()
            p = p.view(*spatial_size, 3).permute(3, 0, 1, 2).contiguous()
            p = p.unsqueeze(0)
            pr = p.expand(x.shape[0], 3, *spatial_size)
        else:
            # compute layers with respect to the input
            fov = FOV_NET_DICT['unet_3_pad{}'.format(padding[0])][n_layers]['fov']
            patch_size = [cs.stop - cs.start for cs in crop_slice_list[0][-3:]]
            fv_shape = x.shape[-3:]
            growth = (np.asarray(patch_size) - fov) / (np.asarray(fv_shape) - 1.0)
            growth = torch.Tensor(growth.tolist()).type(x.type())
            off_input_p = torch.stack([pp * growth + fov / 2 for pp in p])
            nc_list = []
            scan_bound = (torch.Tensor(list(scan_shape[-3:])).type(x.type()) - 1.0)
            for crop_slice in crop_slice_list:
                off_scan = torch.Tensor([cs.start for cs in crop_slice[-3:]]).type(x.type())
                assert ((off_input_p + off_scan - scan_bound <= 0).all().item())
                nc = (off_input_p + off_scan) / scan_bound
                nc -= 0.5
                nc_list.append(nc)
            pr = torch.stack(nc_list)
            pr = pr.view(x.shape[0], *spatial_size, 3).permute(0, 4, 1, 2, 3).contiguous()
        return pr

    def init_graph(self, spatial_size, pool_stride, fv=None):
        n_nodes_total = int(np.prod(spatial_size))
        g = nx.empty_graph(n_nodes_total, create_using=nx.DiGraph)
        for n in range(n_nodes_total):
            s_n = list(np.unravel_index(n, spatial_size))
            cross_n = []
            for d in range(len(spatial_size)):
                for a in list(range(0, spatial_size[d], pool_stride)):
                    temp = s_n[:]
                    temp[d] = a
                    cross_n.append(temp)
            if s_n not in cross_n:
                cross_n.append(s_n)
            end_nodes = np.ravel_multi_index(np.asarray(cross_n).T, spatial_size)
            end_nodes = list(set(end_nodes))
            g.add_edges_from(list(zip(end_nodes, [n] * len(end_nodes))))

        graph = dgl.from_networkx(g)

        return graph.to(fv.device)

    def init(self, initializer):
        initializer.initialize(self)
        nn.init.constant(self.G.weight, 0.0)
        nn.init.constant(self.G.bias, 0.0)

        nn.init.constant(self.r[0].weight, 0.0)
        nn.init.constant(self.r[0].bias, 0.0)
        nn.init.constant(self.theta.weight, 0.0)
        nn.init.constant(self.theta.bias, 0.0)
        nn.init.constant(self.phi.weight, 0.0)
        nn.init.constant(self.phi.bias, 0.0)
        nn.init.constant(self.geo_theta.weight, 0.0)
        nn.init.constant(self.geo_theta.bias, 0.0)
        nn.init.constant(self.geo_phi.weight, 0.0)
        nn.init.constant(self.geo_phi.bias, 0.0)

    def compute_cross_x(self, e_xs, n_x, e_ps, n_p):
        # e_xs: node_batchs, edge_batches, batches, channels
        # n_x: node_batchs, batches, channels
        # e_ps: node_batchs, edge_batches, batches, 3
        # n_p: node_batchs, batches, 3
        node_batches = e_xs.shape[0]
        edge_batches = e_xs.shape[1]
        batches = e_xs.shape[2]
        in_ch = e_xs.shape[3]

        # appearance term
        x_phi = self.phi(e_xs.view(-1, in_ch)).reshape(node_batches, edge_batches, batches, self.d_sim)
        x_theta = self.theta(n_x.view(-1, in_ch)).reshape(node_batches, batches, self.d_sim).unsqueeze(1)
        x_theta = x_theta.permute(2, 0, 1, 3).contiguous()  # [batches, node_batches, 1, d_sim]
        x_phi = x_phi.permute(2, 0, 3, 1).contiguous()  # [batches, node_batches, d_sim, edge_batches]
        f = torch.matmul(x_theta, x_phi)  # [batches, node_batches, 1, edge_batches]
        if self.do_norm:
            f = l2norm(f, dim=1)
        p_phi = self.geo_phi(e_ps.view(-1, 3)).reshape(node_batches, edge_batches, batches, self.p_sim)
        p_theta = self.geo_theta(n_p.view(-1, 3)).reshape(node_batches, batches, self.p_sim).unsqueeze(1)
        p_theta = p_theta.permute(2, 0, 1, 3).contiguous()  # [batches, node_batches, 1, p_sim]
        p_phi = p_phi.permute(2, 0, 3, 1).contiguous()  # [batches, node_batches, p_sim, edge_batches]
        f_p = torch.matmul(p_theta, p_phi)  # [batches, node_batches, 1, edge_batches]
        if self.do_norm:
            f_p = l2norm(f_p, dim=1)
        f_sm = non_local_merge(f + F.relu(f_p), self.merge_method)
        x_g = self.G(e_xs.view(-1, self.in_ch)).reshape(node_batches, edge_batches, batches, self.g_dim)
        x_g = x_g.permute(2, 0, 1, 3).contiguous()  # [batches, node_batches, edge_batches, g_dim]
        y = torch.matmul(f_sm, x_g)  # x_g is [batches, node_batches, 1, self.g_dim]
        cross_x = self.r(y.view(-1, self.g_dim)).reshape(batches, node_batches, in_ch)
        cross_x = cross_x.permute(1, 0, 2).contiguous()  # [node_batches, batches, in_ch]
        return cross_x

    def message_func(self, edges):
        e_xs = edges.src['x']
        e_ps = edges.src['p']
        return {'e_xs': e_xs, 'e_ps': e_ps}

    def reduce_func(self, nodes):
        e_xs = nodes.mailbox['e_xs']
        n_x = nodes.data['x']
        e_ps = nodes.mailbox['e_ps']
        n_p = nodes.data['p']
        cross_x = self.compute_cross_x(e_xs, n_x, e_ps, n_p)
        return {'cross_x': cross_x}

    def plot_attention_map(self, x, crop_slice_list, scan_shape, n_layers, padding, plot_key):
        pass

    def forward(self, x, args=None):
        spatial_sizes = x.shape[-3:]
        h = x
        uid = uuid.uuid4()
        if args is not None:
            crop_slice_list = args[0]
            scan_shape = args[1]
            n_layers = args[2]
            padding = args[3]
        else:
            crop_slice_list = None
            scan_shape = None
            n_layers = None
            padding = None
        if self.plot_path is not None:
            self.plot_attention_map(x, crop_slice_list, scan_shape, n_layers, padding, "{}".format(uid))
        g = self.init_graph(spatial_sizes, 1, fv=h)
        g.to(x.device)
        p = self.build_geo_feature(h, crop_slice_list, scan_shape, n_layers, padding, normalize=True)
        p_flat = p.view(*p.shape[:-3], -1)
        p_flat = p_flat.permute(2, 0, 1).contiguous()
        g.ndata['p'] = p_flat
        for i, pool_stride in enumerate(self.pool_strides):
            h_flat = h.view(*h.shape[:-3], -1)
            h_flat = h_flat.permute(2, 0, 1).contiguous()
            g.ndata['x'] = h_flat
            g.update_all(self.message_func, self.reduce_func)
            cross_x = g.ndata.pop('cross_x')
            cross_x = cross_x.permute(1, 2, 0).contiguous()
            cross_x = cross_x.view(x.shape)
            if self.group_norm:
                h = h + self.bns[i](cross_x)
            else:
                h = h + cross_x
            if self.plot_path is not None:
                self.plot_gradients(cross_x, x, "{}_{}".format(uid, i))
            g.ndata.pop('x')

        self.cn_num += 1
        return F.relu(self.bn(h), inplace=True)


class ConvPoolBlock5d(nn.Module):

    def __init__(self, in_ch_list, base_ch_list, checkpoint_segments,
                 conv_ksize, conv_bias, conv_pad,
                 pool_ksize, pool_strides, pool_pad, dropout=0.1,
                 conv_strdes=1, norm_method='bn', act_method="relu",
                 **kwargs):
        super(ConvPoolBlock5d, self).__init__()
        self.checkpoint_segments = checkpoint_segments
        if dropout > 0:
            self.conv_blocks = nn.Sequential(
                *[nn.Sequential(nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize, stride=conv_strdes,
                                          padding=conv_pad, bias=conv_bias),
                                normal_wrapper(norm_method, base_ch),
                                act_wrapper(act_method),
                                nn.Dropout(dropout))
                  for in_ch, base_ch in zip(in_ch_list, base_ch_list)])
        else:
            self.conv_blocks = nn.Sequential(
                *[nn.Sequential(nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize, stride=conv_strdes,
                                          padding=conv_pad, bias=conv_bias),
                                normal_wrapper(norm_method, base_ch),
                                act_wrapper(act_method)
                                )
                  for in_ch, base_ch in zip(in_ch_list, base_ch_list)])
        self.maxpool = nn.MaxPool3d(kernel_size=pool_ksize, stride=pool_strides, padding=pool_pad)

    def forward(self, x, args=None):
        y = self.conv_blocks(x)
        pooled = self.maxpool(y)
        return y, pooled


class UpsampleConvBlock5d(nn.Module):

    def __init__(self, in_chs, base_chs, checkpoint_segments, scale_factor,
                 conv_ksize, conv_bias, conv_pad, dropout=0.1,
                 norm_method='bn', act_methpd='relu', **kwargs):
        super(UpsampleConvBlock5d, self).__init__()
        self.checkpoint_segments = checkpoint_segments
        self.scale_factor = scale_factor
        if dropout > 0:
            self.conv_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize, padding=conv_pad, bias=conv_bias),
                    normal_wrapper(norm_method, base_ch),
                    act_wrapper(act_methpd),
                    nn.Dropout(dropout)
                ) for in_ch, base_ch in zip(in_chs, base_chs)
            ])
        else:
            self.conv_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize, padding=conv_pad, bias=conv_bias),
                    normal_wrapper(norm_method, base_ch),
                    act_wrapper(act_methpd),
                ) for in_ch, base_ch in zip(in_chs, base_chs)
            ])

        self.merge_func = kwargs.get('merge_func', crop_concat_5d)

    def forward(self, inputs, cats, args=None):
        up_inputs = F.interpolate(inputs, size=None, scale_factor=self.scale_factor,
                                  mode='trilinear', align_corners=True, recompute_scale_factor=True)
        x = crop_concat_5d(up_inputs, cats)
        x = checkpoint(self.conv_blocks, x)
        return x


class ConvBlock5d(nn.Module):

    def __init__(self, in_chs, base_chs, checkpoint_segments, conv_ksize,
                 conv_bias, conv_pad, dropout=0.1, conv_strides=[1, 1],
                 norm_method='bn', act_methpd='relu', lite=False,
                 **kwargs):
        super(ConvBlock5d, self).__init__()
        if lite:
            self.conv_blocks = nn.Sequential(*[
                nn.Sequential(
                    nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize, padding=conv_pad,
                              bias=conv_bias, stride=conv_stride),
                    act_wrapper(act_methpd),
                ) for in_ch, base_ch, conv_stride in zip(in_chs, base_chs, conv_strides)
            ])
        else:
            if dropout > 0:
                self.conv_blocks = nn.Sequential(*[
                    nn.Sequential(
                        nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize, padding=conv_pad,
                                  bias=conv_bias, stride=conv_stride),
                        normal_wrapper(norm_method, base_ch),
                        act_wrapper(act_methpd),
                        nn.Dropout(dropout),
                    ) for in_ch, base_ch, conv_stride in zip(in_chs, base_chs, conv_strides)
                ])
            else:
                self.conv_blocks = nn.Sequential(*[
                    nn.Sequential(
                        nn.Conv3d(in_ch, base_ch, kernel_size=conv_ksize, padding=conv_pad,
                                  bias=conv_bias, stride=conv_stride),
                        normal_wrapper(norm_method, base_ch),
                        act_wrapper(act_methpd),
                    ) for in_ch, base_ch, conv_stride in zip(in_chs, base_chs, conv_strides)
                ])

    def forward(self, x, args=None):
        return self.conv_blocks(x)


class Unet3DCrossBiGeneric(nn.Module):

    def __init__(self, n_layers, in_ch_list, base_ch_list, end_ch_list, out_chs, padding_list,
                 checkpoint_layers, dropout, d_dim, d_sim, p_sim, pool_strides=[1, 1, 1],
                 upsample_ksize=3, upsample_sf=2, non_local_drop_rate=0.0, group_norm=True, do_norm=True,
                 input_spacing=None, input_res=None, norm_method='bn',
                 act_method='relu', merge_method="softmax"):
        super(Unet3DCrossBiGeneric, self).__init__()

        self.dropout = dropout
        self.n_layers = n_layers
        self.d_dim = d_dim
        self.d_sim = d_sim
        self.p_sim = p_sim
        self.in_ch_list = in_ch_list
        self.base_ch_list = base_ch_list
        self.end_ch_list = end_ch_list
        self.upsample_ksize = upsample_ksize
        self.upsample_sf = upsample_sf
        self.padding_list = padding_list
        self.non_local_drop_rate = non_local_drop_rate
        self.input_res = input_res
        self.input_spacing = input_spacing
        self.checkpoint_layers = checkpoint_layers
        self.pool_strides = pool_strides
        self.merge_method = merge_method
        self.group_norm = group_norm
        self.do_norm = do_norm
        # self.contextual_layers = contextual_layers
        assert (len(end_ch_list) == len(base_ch_list) == len(in_ch_list) == len(padding_list))
        self.out_chs = out_chs
        self.ds_modules = nn.ModuleList(
            [
                ConvPoolBlock5d([in_ch_list[n], base_ch_list[n]], [base_ch_list[n],
                                                                   end_ch_list[n]],
                                checkpoint_layers[n], 3, False, padding_list[n],
                                2, 2, 0, norm_method=norm_method, act_method=act_method, dropout=dropout)
                for n in range(n_layers)
            ]
        )

        self.bg = ConvBlock5d([in_ch_list[n_layers], base_ch_list[n_layers]],
                              [base_ch_list[n_layers], end_ch_list[n_layers]],
                              checkpoint_layers[n_layers], 3, False, padding_list[n_layers],
                              dropout, norm_method=norm_method,
                              act_method=act_method, conv_strides=[1, 1])

        self.middle_psa = CBNet(end_ch_list[n_layers], self.d_dim,
                                            self.d_sim, self.p_sim
                                            , pool_strides=self.pool_strides,
                                            drop_rate=self.non_local_drop_rate,
                                            merge_method=self.merge_method,
                                            group_norm=group_norm, do_norm=self.do_norm)
        self.us_modules = nn.ModuleList(
            [
                UpsampleConvBlock5d([in_ch_list[n_layers + 1 + n],
                                     base_ch_list[n_layers + 1 + n]],
                                    [base_ch_list[n_layers + 1 + n],
                                     end_ch_list[n_layers + 1 + n]],
                                    checkpoint_layers[n_layers + 1 + n], self.upsample_sf,
                                    self.upsample_ksize, False, padding_list[n_layers + 1 + n],
                                    norm_method=norm_method, act_method=act_method, dropout=dropout)
                for n in range(n_layers)
            ]
        )
        self.dummy = torch.ones(1, requires_grad=True)
        self.outs = nn.ModuleList(
            [nn.Conv3d(end_ch_list[-1], out_ch, kernel_size=1) for out_ch in self.out_chs]
        )

    def finetune(self):
        set_trainable(self.outs, False)
        set_trainable(self.us_modules, False)
        set_trainable(self.bg, False)
        set_trainable(self.ds_modules, False)
        self.middle_psa.finetune()

    def set_trace_path(self, *args):
        path = args[0]
        data = args[1]
        self.middle_psa.plot_path = path
        self.middle_psa.extra_data = data

    def init(self, initializer):
        initializer.initialize(self)

    def forward(self, x, args=None):
        if x.type() != self.ds_modules[0].conv_blocks[0][0].weight.type():
            previous_t = x.type()
            x = x.type(self.ds_modules[0].conv_blocks[0][0].weight.type())
        else:
            previous_t = None
        ds_feat_list = [(x,)]
        for idx, ds in enumerate(self.ds_modules):
            if self.checkpoint_layers[idx] > 0:
                if idx == 0:
                    ds_feat_list.append(checkpoint(ds, ds_feat_list[-1][-1], self.dummy))
                else:
                    ds_feat_list.append(checkpoint(ds, ds_feat_list[-1][-1]))
            else:
                ds_feat_list.append(ds(ds_feat_list[-1][-1]))
        ds_feat_list.pop(0)
        xbg = checkpoint_wrapper(self.bg, self.checkpoint_layers[self.n_layers], ds_feat_list[-1][-1])
        if args is not None:
            pargs = args + (self.n_layers, self.padding_list[self.n_layers])
        else:
            pargs = None
        xbg = self.middle_psa(xbg, pargs)
        us_feat_list = [xbg]
        for us, ds_feat in zip(self.us_modules, reversed(ds_feat_list)):
            us_feat_list.append(checkpoint(us, us_feat_list[-1], ds_feat[0]))
        end_f = us_feat_list[-1]
        o_list = [out(end_f) for out in self.outs]

        if previous_t is not None:
            o_list = [o.type(previous_t) for o in o_list]
        return o_list


class Unet3DCrossLocalGeneric(nn.Module):

    def __init__(self, n_layers, in_ch_list, base_ch_list, end_ch_list, out_chs, padding_list,
                 checkpoint_layers, dropout, non_local_dim, non_local_sim,
                 non_local_layer, pool_strides=[1, 1, 1],
                 non_local_drop_rate=0.0, merge_method='softmax', group_norm=True,
                 upsample_ksize=3, upsample_sf=2,
                 input_spacing=None, input_res=None,
                 norm_method='bn', act_method='relu'):
        super(Unet3DCrossLocalGeneric, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.non_local_dim = non_local_dim
        self.non_local_sim = non_local_sim
        self.non_local_layer = non_local_layer
        self.pool_strides = pool_strides
        self.padding_list = padding_list
        self.merge_method = merge_method
        self.group_norm = group_norm
        self.non_local_drop_rate = non_local_drop_rate
        self.in_ch_list = in_ch_list
        self.base_ch_list = base_ch_list
        self.end_ch_list = end_ch_list
        self.upsample_ksize = upsample_ksize
        self.upsample_sf = upsample_sf
        self.input_res = input_res
        self.input_spacing = input_spacing
        self.checkpoint_layers = checkpoint_layers
        # self.contextual_layers = contextual_layers
        assert (len(end_ch_list) == len(base_ch_list) == len(in_ch_list) == len(padding_list))
        self.out_chs = out_chs
        self.ds_modules = nn.ModuleList(
            [
                ConvPoolBlock5d([in_ch_list[n], base_ch_list[n]], [base_ch_list[n],
                                                                   end_ch_list[n]],
                                checkpoint_layers[n], 3, False, padding_list[n],
                                2, 2, 0, norm_method=norm_method, act_method=act_method,
                                dropout=dropout)
                for n in range(n_layers)
            ]
        )

        self.bg = ConvBlock5d([in_ch_list[n_layers], base_ch_list[n_layers]],
                              [base_ch_list[n_layers], end_ch_list[n_layers]],
                              checkpoint_layers[n_layers], 3, False, padding_list[n_layers],
                              dropout, norm_method=norm_method,
                              act_method=act_method, conv_strides=[1, 1])

        self.non_local_module = CNet(end_ch_list[n_layers], non_local_dim,
                                             non_local_sim, pool_strides=self.pool_strides,
                                             drop_rate=self.non_local_drop_rate,
                                             merge_method=self.merge_method,
                                         group_norm=group_norm)

        self.us_modules = nn.ModuleList(
            [
                UpsampleConvBlock5d([in_ch_list[n_layers + 1 + n],
                                     base_ch_list[n_layers + 1 + n]],
                                    [base_ch_list[n_layers + 1 + n],
                                     end_ch_list[n_layers + 1 + n]],
                                    checkpoint_layers[n_layers + 1 + n], self.upsample_sf,
                                    self.upsample_ksize, False, padding_list[n_layers + 1 + n],
                                    norm_method=norm_method, act_method=act_method, dropout=dropout)
                for n in range(n_layers)
            ]
        )
        self.dummy = torch.ones(1, requires_grad=True)
        self.outs = nn.ModuleList(
            [nn.Conv3d(end_ch_list[-1], out_ch, kernel_size=1) for out_ch in self.out_chs]
        )

    def finetune(self):
        set_trainable(self.outs, False)
        set_trainable(self.us_modules, False)
        set_trainable(self.bg, False)
        set_trainable(self.ds_modules, False)
        self.non_local_module.finetune()

    def set_trace_path(self, *args):
        path = args[0]
        data = args[1]
        self.non_local_module.plot_path = path
        self.non_local_module.extra_data = data

    def init(self, initializer):
        initializer.initialize(self)

    def forward(self, x, args=None):
        if x.type() != self.ds_modules[0].conv_blocks[0][0].weight.type():
            previous_t = x.type()
            x = x.type(self.ds_modules[0].conv_blocks[0][0].weight.type())
        else:
            previous_t = None

        ds_feat_list = [(x,)]
        for idx, ds in enumerate(self.ds_modules):
            if self.checkpoint_layers[idx] > 0:
                if idx == 0:
                    ds_feat_list.append(checkpoint(ds, ds_feat_list[-1][-1], self.dummy))
                else:
                    ds_feat_list.append(checkpoint(ds, ds_feat_list[-1][-1]))
            else:
                ds_feat_list.append(ds(ds_feat_list[-1][-1]))
        ds_feat_list.pop(0)
        xbg = checkpoint_wrapper(self.bg, self.checkpoint_layers[self.n_layers], ds_feat_list[-1][-1])
        if args is not None:
            pargs = args + (self.n_layers, self.padding_list[self.n_layers])
        else:
            pargs = None
        xbg = self.non_local_module(xbg, pargs)
        us_feat_list = [xbg]
        for us, ds_feat in zip(self.us_modules, reversed(ds_feat_list)):
            us_feat_list.append(checkpoint(us, us_feat_list[-1], ds_feat[0]))
            # us_feat_list.append(us(us_feat_list[-1], ds_feat[0]))
        end_f = us_feat_list[-1]
        o_list = [out(end_f) for out in self.outs]
        if previous_t is not None:
            o_list = [o.type(previous_t) for o in o_list]
        return o_list

class CTSUNet(TSUnet):

    def __init__(self, net1_param, net2_param,
                 ds_rate, norm_method='bn', act_method='relu',
                 return_res=False, training_stage=0):
        super(CTSUNet, self).__init__(None, None,
                                            ds_rate, norm_method, act_method,
                                            return_res, training_stage)
        net1_param['norm_method'] = norm_method
        net2_param['norm_method'] = norm_method
        net1_param['act_method'] = act_method
        net2_param['act_method'] = act_method

        self.unet1 = Unet3DCrossBiGeneric(**net1_param)
        self.unet2 = Unet3DCrossBiGeneric(**net2_param)
        print(net1_param)
        print(net2_param)


class TSUnetCrossBiV3(TSUnet):

    def __init__(self, net1_param, net2_param,
                 ds_rate, norm_method='bn', act_method='relu',
                 return_res=False, training_stage=0):
        super(TSUnetCrossBiV3, self).__init__(None, None,
                                              ds_rate, norm_method, act_method,
                                              return_res, training_stage)
        net1_param['norm_method'] = norm_method
        net2_param['norm_method'] = norm_method
        net1_param['act_method'] = act_method
        net2_param['act_method'] = act_method

        self.unet1 = Unet3DCrossBiGeneric(**net1_param)
        self.unet2 = Unet3DCrossBiGeneric(**net2_param)
        print(net1_param)
        print(net2_param)



