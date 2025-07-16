import os 
import random
import numpy as np
import torch
from PIL import Image
import torchvision
import time
import sys
from sklearn.decomposition import PCA
from networks.autoencoder import Autoencoder
import torch.nn.functional as F
import torch.nn as nn
from utils.pos_embed import interpolate_pos_embed
import scipy.stats as stats
from torchmetrics.functional import structural_similarity_index_measure as ssim
from tqdm import tqdm
from scipy.stats import t
import encoder_model.vit4mae as vit4mae
import encoder_model.vit4cae as vit4cae
import encoder_model.swin4simmim as swin4simmim
import yaml
from mmselfsup.models.backbones import MaskFeatViT, BEiTViT
from mmcls.models import VisionTransformer, BEiT, MixMIMTransformer
import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # print(pred)
        correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))
        # print(correct)

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def set_seed(seed):
    # for reproducibility. 
    # note that pytorch is not completely reproducible 
    # https://pytorch.org/docs/stable/notes/randomness.html  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.initial_seed() # dataloader multi processing 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return None

def save_model(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def build_model(model_name, method=None):
    if model_name == 'vit_base' and (method == 'MAE' or method == 'IBOT'):
        model = vit4mae.__dict__['vit_base_patch16'](
            global_pool=False,
        )
        embed_dim = 768
    elif model_name == 'vit_large' and (method == 'MAE' or method == 'IBOT'):
        model = vit4mae.__dict__['vit_large_patch16'](
            global_pool=False,
        )
        embed_dim = 1024
    elif model_name == 'cae_base':
        model = vit4cae.__dict__['cae_base_patch16_224'](
            use_abs_pos_emb=False,
            init_values=0.1,
        )
        embed_dim = 768
    elif model_name == 'cae_large':
        model = vit4cae.__dict__['cae_large_patch16_224'](
            use_abs_pos_emb=False,
            init_values=0.1,
        )
        embed_dim = 1024
    elif model_name == 'swin_base':
        path = r'D:\Exp\DOV4MIM\encoder_model\simmim_configs\swin_base__800ep\simmim_pretrain__swin_base__img192_window6__800ep.yaml'
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        model = swin4simmim.build_swin(config=config)
        embed_dim = 1024
    elif model_name == 'swin_large':
        path = r'D:\Exp\DOV4MIM\encoder_model\simmim_configs\swin_large__800ep\simmim_pretrain__swin_large__img192_window12__800ep.yaml'
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        model = swin4simmim.build_swin(config=config)
        embed_dim = 1536
    elif model_name == 'vit_base' and (method == 'MaskFeat' or method == 'PixMIM' or method == 'EVA'):
        model = VisionTransformer()
        embed_dim = 768
    elif model_name == 'vit_base' and (method == 'BEIT' or method == 'BEITv2'):
        model = BEiTViT()
        embed_dim = 768
    elif model_name == 'mixmim_base':
        model = MixMIMTransformer()
        embed_dim = 1024

    return model, embed_dim

def load_model(model, path):
    model_path = path
    checkpoint = torch.load(model_path, map_location="cuda")
    print("Load pre-trained checkpoint from: %s" % model_path)

    if 'MAE' in model_path:
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate_pos_embed(model, checkpoint_model)
       
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print('msg:', msg)

    elif 'CAE' in model_path:
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        for key in list(checkpoint_model.keys()):
            if 'encoder.' in key:
                new_key = key.replace('encoder.','')
                checkpoint_model[new_key] = checkpoint_model[key]
                checkpoint_model.pop(key)
            if 'teacher' in key or 'decoder' in key:
                    checkpoint_model.pop(key)

        all_keys = list(checkpoint_model.keys())

        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print('msg:', msg)
    
    elif 'IBOT' in model_path:
        if 'state_dict' in checkpoint.keys():
            checkpoint_model = checkpoint['state_dict']  # pre-trained on ImageNet
        else:
            checkpoint_model = checkpoint['teacher']  # pre-trained on ImageNet-20/50/100
            for key in list(checkpoint_model.keys()):
                if 'backbone.' in key:
                    new_key = key.replace('backbone.','')
                    checkpoint_model[new_key] = checkpoint_model[key]
                    checkpoint_model.pop(key)
        state_dict = model.state_dict()

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print('msg:', msg)
    
    elif 'SimMIM' in model_path:
        checkpoint_model = checkpoint['model']  # pre-trained on ImageNet-20/50/100
        for key in list(checkpoint_model.keys()):
            if 'encoder.' in key:
                new_key = key.replace('encoder.','')
                checkpoint_model[new_key] = checkpoint_model[key]
                checkpoint_model.pop(key)
        state_dict = model.state_dict()

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print('msg:', msg)

    elif 'BEIT' in model_path or 'MaskFeat' in model_path or 'MixMIM' in model_path or 'PixMIM' in model_path:
        checkpoint_model = checkpoint['state_dict']  # pre-trained on ImageNet-20/50/100
        for key in list(checkpoint_model.keys()):
            if 'backbone.' in key:
                new_key = key.replace('backbone.','')
                checkpoint_model[new_key] = checkpoint_model[key]
                checkpoint_model.pop(key)
        state_dict = model.state_dict()

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print('msg:', msg)

    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    
    return model

def cal_Cosinesimilarity(tensor):  # 输入为(bs, feature_dim)
    # 1. 计算每个样本的L2范数
    norms = torch.norm(tensor, dim=1, keepdim=True)

    # 2. 转置张量
    tensor_transposed = tensor.t()

    # 3. 计算每个样本之间的点积
    dot_products = torch.matmul(tensor, tensor_transposed)

    # 4. 计算余弦相似度矩阵
    cosine_similarity = dot_products / (norms * norms.t())

    cosine_similarity = torch.clamp(cosine_similarity, 0, 1)

    # 计算两两样本相似度的平均值（不包括自己和自己的相似度）
    k = tensor.size(0)
    n = k*(k-1)/2
    s = (torch.sum(cosine_similarity)-k)/2
    mean_cos = s/n

    return cosine_similarity, mean_cos

def cal_Cosinesimilarity(tensor):  # 输入为(bs, feature_dim)
    # 1. 计算每个样本的L2范数
    norms = torch.norm(tensor, dim=1, keepdim=True)

    # 2. 转置张量
    tensor_transposed = tensor.t()

    # 3. 计算每个样本之间的点积
    dot_products = torch.matmul(tensor, tensor_transposed)

    # 4. 计算余弦相似度矩阵
    cosine_similarity = dot_products / (norms * norms.t())

    cosine_similarity = torch.clamp(cosine_similarity, 0, 1)

    # 计算两两样本相似度的平均值（不包括自己和自己的相似度）
    k = tensor.size(0)
    n = k*(k-1)/2
    s = (torch.sum(cosine_similarity)-k)/2
    mean_cos = s/n

    return cosine_similarity, mean_cos

def cal_euclidean_similarity(tensor, sigma=1.0):
    m, n = tensor.size()
    euclidean_similarity = torch.zeros((m, m))

    for i in range(m):
        for j in range(i, m):
            distance = torch.norm(tensor[i] - tensor[j])
            # RBF kernel similarity: S = exp(-D^2 / (2 * σ^2))
            similarities = torch.exp(-distance**2 / (2 * sigma**2))
            euclidean_similarity[i, j] = similarities
            euclidean_similarity[j, i] = similarities  # 对称性

    # 计算两两样本相似度的平均值（不包括自己和自己的相似度）
    k = tensor.size(0)
    n = k*(k-1)/2
    s = (torch.sum(euclidean_similarity)-k)/2
    mean_cos = s/n
    
    return euclidean_similarity, mean_cos

def pca_tensor(input, k):
    data_mean = torch.mean(input, dim=0)
    data_std = torch.std(input, dim=0)
    data_normalized = torch.zeros_like(input)
    for i in range(data_normalized.shape[0]):
        if data_std[i] != 0:
            data_normalized[i] = (input[i] - data_mean[i]) / data_std[i]

    pca = PCA(n_components=k)  # 降维到 k 维
    data_normalized = pca.fit_transform(data_normalized.numpy())
    data_normalized = torch.tensor(data_normalized, dtype=torch.float)
    
    return data_normalized

def create_edge_index(n):
    # Creating the first row
    row1 = torch.arange(n, dtype=torch.long).repeat_interleave(n)
    # Creating the second row
    row2 = torch.arange(n, dtype=torch.long).repeat(n)
    # Combining the two rows
    tensor = torch.stack([row1, row2])
    # tensor = torch.tensor(tensor, dtype=torch.long)
    return tensor

def auto_encoder_tensor(input, weight, k):

    autoencoder = Autoencoder(input.size(1), k)
    checkpoint = torch.load(weight, map_location='cuda:0')
    autoencoder.load_state_dict(checkpoint['state_dict'])
    autoencoder = autoencoder.cuda()
    output = autoencoder.encoder(input)
    
    return output

class RandomPatchMasking(nn.Module):
    def __init__(self, patch_size=16, min_mask_ratio=0.25, max_mask_ratio=0.75, mask_value=0.5, random=False, need_mask=False):
        super().__init__()
        self.patch_size = patch_size
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.mask_value = mask_value
        self.random = random
        self.need_mask = need_mask

    def forward(self, x):
        # 确认每个维度的patch数量
        RGB_INPUT = False
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
            RGB_INPUT = True
        _, _, height, width = x.shape
        num_patches_along_height = height // self.patch_size
        num_patches_along_width = width // self.patch_size

        # 使用unfold方法分割张量到patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # print(patches.shape)
        patches = patches.contiguous().view(*patches.shape[:4], -1)
        # print(patches.shape)

        # 计算总共有多少patches
        total_patches = num_patches_along_height * num_patches_along_width
        # 计算需要掩码的patches数量
        if self.random:
            rand_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)
            num_mask_patches = int(rand_ratio * total_patches)
        else:
            num_mask_patches = int(self.max_mask_ratio * total_patches)

        # 生成一个随机的索引序列
        # l = torch.randperm(total_patches, device=x.device)
        l = torch.randperm(total_patches)
        mask_pos = l[:num_mask_patches]
        unmask_pos = l[num_mask_patches:]
        # print('mask_pos:', len(mask_pos))
        # 对选中的patches应用掩码
        # print('patches:', patches.shape)
        batch_size, num_channels, _, _, patch_size_squared = patches.shape
        patches.view(batch_size, num_channels, total_patches, patch_size_squared)[:, :, mask_pos, :] = self.mask_value
        # print('patches:', patches.view(batch_size, num_channels, total_patches, patch_size_squared).shape)

        # 将patches重构回原来的图像形状
        reconstructed = patches.view(batch_size, num_channels, num_patches_along_height, num_patches_along_width, self.patch_size, self.patch_size)
        reconstructed = reconstructed.permute(0, 1, 2, 4, 3, 5).contiguous().view(batch_size, num_channels, height, width)
        # print('reconstructed:', reconstructed.shape)

        if RGB_INPUT:
            reconstructed = torch.squeeze(reconstructed, dim=0)

        if self.need_mask:
            return reconstructed, mask_pos, unmask_pos
        else:
            return reconstructed

    def __call__(self, x):
        return self.forward(x)
    
class MultiRandomPatchMasking(nn.Module):
    def __init__(self, patch_size=16, need_mask=False):
        super().__init__()
        self.patch_size = patch_size
        self.need_mask = need_mask

    def forward(self, x1, x2):
        _, _, height_1, width_1 = x1.shape
        _, _, height_2, width_2 = x2.shape

        assert height_1 == height_2 and width_1 == width_2

        num_patches_along_height = height_1 // self.patch_size
        num_patches_along_width = width_1 // self.patch_size

        # 使用unfold方法分割张量到patches
        patches_1 = x1.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # print(patches.shape)
        patches_1 = patches_1.contiguous().view(*patches_1.shape[:4], -1)
        # print(patches.shape)

        patches_2 = x2.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # print(patches.shape)
        patches_2 = patches_2.contiguous().view(*patches_2.shape[:4], -1)

        # 计算总共有多少patches
        total_patches = num_patches_along_height * num_patches_along_width
        # 计算需要掩码的patches数量

        num_mask_patches = int(0.5 * total_patches)

        # 生成一个随机的索引序列
        rand_pos = torch.randperm(total_patches, device=x1.device)
        mask_pos_1 = rand_pos[:num_mask_patches]
        mask_pos_2 = rand_pos[num_mask_patches:]
        # print('mask_pos:', len(mask_pos))
        # 对选中的patches应用掩码
        # print('patches:', patches.shape)
        batch_size, num_channels, _, _, patch_size_squared = patches_1.shape
        patches_1.view(batch_size, num_channels, total_patches, patch_size_squared)[:, :, mask_pos_1, :] = patches_2.view(batch_size, num_channels, total_patches, patch_size_squared)[:, :, mask_pos_1, :]
        # print('patches:', patches.view(batch_size, num_channels, total_patches, patch_size_squared).shape)

        # 将patches重构回原来的图像形状
        reconstructed = patches_1.view(batch_size, num_channels, num_patches_along_height, num_patches_along_width, self.patch_size, self.patch_size)
        reconstructed = reconstructed.permute(0, 1, 2, 4, 3, 5).contiguous().view(batch_size, num_channels, height_1, width_1)
        # print('reconstructed:', reconstructed.shape)

        if self.need_mask:
            return reconstructed, mask_pos_1
        else:
            return reconstructed

    def __call__(self, x):
        return self.forward(x)
    
class CenterMasking(nn.Module):
    def __init__(self, patch_size=16, min_mask_ratio=0.25, max_mask_ratio=0.75, mask_value=0):
        super().__init__()
        self.patch_size = patch_size
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.mask_value = mask_value

    def forward(self, x):
        # 确认每个维度的patch数量
        RGB_INPUT = False
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=0)
            RGB_INPUT = True
        _, _, height, width = x.shape
        num_patches_along_height = height // self.patch_size
        num_patches_along_width = width // self.patch_size

        reconstructed = x.clone()
        # print('reconstructed:', reconstructed.shape)
        reconstructed[:, :, 1*self.patch_size:(num_patches_along_height-1)*self.patch_size, 1*self.patch_size:(num_patches_along_width-1)*self.patch_size] = self.mask_value

        if RGB_INPUT:
            reconstructed = torch.squeeze(reconstructed, dim=0)

        return reconstructed

    def __call__(self, x):
        return self.forward(x)  

def one_tailed_ttest(res_sim_adv, res_sim_shadow):
    t_stat, p_val = stats.ttest_ind(res_sim_adv, res_sim_shadow)
    # t_stat, p_val = stats.ttest_rel(res_sim_adv, res_sim_shadow)

    if t_stat > 0:
        p_val_one_sided = p_val / 2
    else:
        p_val_one_sided = 1 - p_val / 2
    
    return t_stat, p_val_one_sided

def batch_pairwise_ssim(batch1, batch2):
    batch1_size = batch1.size(0)
    batch2_size = batch2.size(0)
    ssim_matrix = torch.zeros(batch1_size, batch2_size)

    for i in range(batch1_size):
        # 批量计算 batch1[i] 与 batch2 的 SSIM
        ssim_matrix[i] = ssim(batch1[i].unsqueeze(0).expand(batch2_size, -1, -1, -1), batch2, data_range=1.0)
    print('max:', torch.max(ssim_matrix[0]))

    return ssim_matrix

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func) 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_entropy(tensor):
    # 对每一行进行归一化，得到概率分布
    tensor_normalized = F.softmax(tensor, dim=1)  # 使用 softmax 将每行元素转换为概率分布
    
    # 计算每行的熵
    entropy = -torch.sum(tensor_normalized * torch.log(tensor_normalized + 1e-10), dim=1)  # 计算熵, 加上小的epsilon避免log(0)
    
    return entropy

def mutual_information(a, b, bins=100):
    # 计算 a 和 b 的联合分布
    joint_hist = torch.histc(a, bins=bins, min=a.min(), max=a.max()) + torch.histc(b, bins=bins, min=b.min(), max=b.max())
    
    # 归一化联合分布
    joint_prob = joint_hist / joint_hist.sum()
    
    # 计算 a 和 b 的边缘分布
    p_a = torch.histc(a, bins=bins, min=a.min(), max=a.max())
    p_b = torch.histc(b, bins=bins, min=b.min(), max=b.max())
    
    # 归一化边缘分布
    p_a = p_a / p_a.sum()
    p_b = p_b / p_b.sum()
    
    # 计算互信息
    mi = torch.sum(joint_prob * torch.log(joint_prob / (p_a.unsqueeze(0) * p_b.unsqueeze(1) + 1e-10)))
    
    return mi

def grubbs_test(data, value, alpha=0.05, mode='high'):
    n = len(data)
    mean_y = np.mean(data)
    std_y = np.std(data, ddof=1)
    
    # 计算G统计量
    if mode == 'high':
        G = (value - mean_y) / std_y
    elif mode == 'low':
        G = (mean_y - value) / std_y
    
    # 计算临界值
    critical_value = ((n - 1) / np.sqrt(n)) * np.sqrt((t.ppf(1 - alpha / (2 * n), n - 2)**2) / (n - 2 + t.ppf(1 - alpha / (2 * n), n - 2)**2))
    
    return G > critical_value

def one_tailed_ttest(res_sim_adv, res_sim_shadow):
    t_stat, p_val = stats.ttest_ind(res_sim_adv, res_sim_shadow)
    # t_stat, p_val = stats.ttest_rel(res_sim_adv, res_sim_shadow)

    if t_stat > 0:
        p_val_one_sided = p_val / 2
    else:
        p_val_one_sided = 1 - p_val / 2
    
    return t_stat, p_val_one_sided

def generate_gassian_ditribution(N, length):
    noise = torch.normal(0.0, 1.0, size=(N,length)) 
    noise = noise.sort(dim=1)[0]
    # print(noise.shape)
    # print(noise)
    noise = (-(noise-noise.mean(dim=1, keepdim=True))**2/(2*noise.std(dim=1,  keepdim=True)**2)).exp()/(math.sqrt(2*math.pi)*noise.std(dim=1,  keepdim=True))
    noise = noise/noise.sum(dim=1, keepdim=True)
    # print(noise)
    return noise

def obtain_membership_feature(global_feature_map, local_feature_vectors, feature_type='both'):
    ## feature calculation to obtain the score distribution 
    B, L, D = global_feature_map.shape

    assert local_feature_vectors.shape[0]%global_feature_map.shape[0]==0
    local_feature_vectors = local_feature_vectors.view(B, -1, D)

    sim_score = torch.einsum('nik,njk->nij',[local_feature_vectors, global_feature_map])  # B N L
    assert sim_score.shape[1:]==(local_feature_vectors.shape[1], global_feature_map.shape[1])
    logit_score = F.log_softmax(sim_score,dim=2)
    
    _, N, L = sim_score.shape

    if feature_type=='both':
        ## calculate the engery with uniform distribution
        uniform = torch.ones_like(sim_score).to(sim_score.device)/L
        uniform_score = (uniform * ((uniform+1e-6).log()-logit_score)).sum(dim=2) # rank 
        sorted_uniform_score = torch.sort(uniform_score, dim=1, descending=True)[0] # B N

        ## calculate the engery with gassian distribution
        gassian = generate_gassian_ditribution(N,L).to(sim_score.device).unsqueeze(dim=0).repeat(B,1,1)
        gassian_score = (gassian * ((gassian+1e-6).log()-logit_score)).sum(dim=2) 
        sorted_gassian_score = torch.sort(gassian_score, dim=1, descending=True)[0]  # B N

        feature = torch.cat([sorted_uniform_score, sorted_gassian_score], dim=1)  # B 2N

        return feature

    elif feature_type=='uniform':
        ## calculate the engery with uniform distribution
        uniform = torch.ones_like(sim_score).to(sim_score.device)/L
        uniform_score = (uniform * ((uniform+1e-6).log()-logit_score)).sum(dim=2) # rank 
        sorted_uniform_score = torch.sort(uniform_score, dim=1, descending=True)[0] # B N

        return sorted_uniform_score
    
    elif feature_type=='gassian':
        gassian = generate_gassian_ditribution(N,L).to(sim_score.device).unsqueeze(dim=0).repeat(B,1,1)
        gassian_score = (gassian * ((gassian+1e-6).log()-logit_score)).sum(dim=2) 
        sorted_gassian_score = torch.sort(gassian_score, dim=1, descending=True)[0]  # B N

        return sorted_gassian_score

    else:
        NotImplementedError()


if __name__=='__main__':
    t = CenterMasking(patch_size=1, mask_value=0)

    img = torch.rand(1,3,6,6)
    output = t(img)
    print(output)