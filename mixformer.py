'''
mixformer模型代码
'''
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, Mlp, trunc_normal_

from lib.utils.misc import is_main_process
from lib.models.mixformer_cvt.head import build_box_head
from lib.models.mixformer_cvt.utils import to_2tuple
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from lib.models.mixformer_vit.pos_utils import get_2d_sincos_pos_embed


class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x, t_h, t_w, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q_mt, q_s = torch.split(q, [t_h * t_w * 2, s_h * s_w], dim=2)
        k_mt, k_s = torch.split(k, [t_h * t_w * 2, s_h * s_w], dim=2)
        v_mt, v_s = torch.split(v, [t_h * t_w * 2, s_h * s_w], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, t_h*t_w*2, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_test(self, x, s_h, s_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv_s = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_s, _, _ = qkv_s.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        qkv = torch.cat([self.qkv_mem, qkv_s], dim=3)
        _, k, v = qkv.unbind(0)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, s_h*s_w, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def set_online(self, x, t_h, t_w):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        self.qkv_mem = qkv
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple) [B, num_heads, N, C//num_heads]

        # asymmetric mixed attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, t_h, t_w, s_h, s_w):
        x = x + self.drop_path1(self.attn(self.norm1(x), t_h, t_w, s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

    def forward_test(self, x, s_h, s_w):
        x = x + self.drop_path1(self.attn.forward_test(self.norm1(x), s_h, s_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

    def set_online(self, x, t_h, t_w):
        x = x + self.drop_path1(self.attn.set_online(self.norm1(x), t_h, t_w))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class CBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#        self.attn = nn.Conv2d(dim, dim, 13, padding=6, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.drop_path(self.conv2(self.attn(mask * self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        return x


class ConvViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size_s=288, img_size_t=128, patch_size=[4, 2, 2], embed_dim=[256, 384, 768],
                 depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], in_chans=3, num_classes=1000,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed1 = PatchEmbed(
            patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + i], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + depth[1] + i], norm_layer=norm_layer)
            for i in range(depth[2])])

        self.norm = norm_layer(embed_dim[-1])

        self.apply(self._init_weights)

        self.grid_size_s = img_size_s // (patch_size[0] * patch_size[1] * patch_size[2])
        self.grid_size_t = img_size_t // (patch_size[0] * patch_size[1] * patch_size[2])
        self.num_patches_s = self.grid_size_s ** 2
        self.num_patches_t = self.grid_size_t ** 2
        self.pos_embed_s = nn.Parameter(torch.zeros(1, self.num_patches_s, embed_dim[2]), requires_grad=False)
        self.pos_embed_t = nn.Parameter(torch.zeros(1, self.num_patches_t, embed_dim[2]), requires_grad=False)

        self.init_pos_embed()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def init_pos_embed(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_t = get_2d_sincos_pos_embed(self.pos_embed_t.shape[-1], int(self.num_patches_t ** .5),
                                              cls_token=False)
        self.pos_embed_t.data.copy_(torch.from_numpy(pos_embed_t).float().unsqueeze(0))

        pos_embed_s = get_2d_sincos_pos_embed(self.pos_embed_s.shape[-1], int(self.num_patches_s ** .5),
                                              cls_token=False)
        self.pos_embed_s.data.copy_(torch.from_numpy(pos_embed_s).float().unsqueeze(0))

    def forward(self, x_t, x_ot, x_s):
        """
        :param x_t: (batch, c, 128, 128)
        :param x_s: (batch, c, 288, 288)
        :return:
        """
        ### conv embeddings for x_t
        x_t = self.patch_embed1(x_t)
        x_t = self.pos_drop(x_t)
        for blk in self.blocks1:
            x_t = blk(x_t)
        x_t = self.patch_embed2(x_t)
        for blk in self.blocks2:
            x_t = blk(x_t)
        x_t = self.patch_embed3(x_t)
        x_t = x_t.flatten(2).permute(0, 2, 1) #BCHW --> BNC
        x_t = self.patch_embed4(x_t)

        ### conv embeddings for x_ot
        x_ot = self.patch_embed1(x_ot)
        x_ot = self.pos_drop(x_ot)
        for blk in self.blocks1:
            x_ot = blk(x_ot)
        x_ot = self.patch_embed2(x_ot)
        for blk in self.blocks2:
            x_ot = blk(x_ot)
        x_ot = self.patch_embed3(x_ot)
        x_ot = x_ot.flatten(2).permute(0, 2, 1)
        x_ot = self.patch_embed4(x_ot)

        ### conv embeddings for x_s
        x_s = self.patch_embed1(x_s)
        x_s = self.pos_drop(x_s)
        for blk in self.blocks1:
            x_s = blk(x_s)
        x_s = self.patch_embed2(x_s)
        for blk in self.blocks2:
            x_s = blk(x_s)
        x_s = self.patch_embed3(x_s)
        x_s = x_s.flatten(2).permute(0, 2, 1)
        x_s = self.patch_embed4(x_s)

        B, C = x_t.size(0), x_t.size(-1)
        H_s = W_s = self.grid_size_s
        H_t = W_t = self.grid_size_t

        x_s = x_s + self.pos_embed_s
        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x = torch.cat([x_t, x_ot, x_s], dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks3:
            x = blk(x, H_t, W_t, H_s, W_s)

        x_t, x_ot, x_s = torch.split(x, [H_t * W_t, H_t * W_t, H_s * W_s], dim=1)

        x_t_2d = x_t.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_ot_2d = x_ot.transpose(1, 2).reshape(B, C, H_t, W_t)
        x_s_2d = x_s.transpose(1, 2).reshape(B, C, H_s, W_s)

        return x_t_2d, x_ot_2d, x_s_2d

    def forward_test(self, x_s):
        x_s = self.patch_embed1(x_s)
        x_s = self.pos_drop(x_s)
        for blk in self.blocks1:
            x_s = blk(x_s)
        x_s = self.patch_embed2(x_s)
        for blk in self.blocks2:
            x_s = blk(x_s)
        x_s = self.patch_embed3(x_s)
        x_s = x_s.flatten(2).permute(0, 2, 1)
        x_s = self.patch_embed4(x_s)

        H_s = W_s = self.grid_size_s
        x_s = x_s + self.pos_embed_s
        x_s = self.pos_drop(x_s)

        for blk in self.blocks3:
            x_s = blk.forward_test(x_s, H_s, W_s)

        x_s = rearrange(x_s, 'b (h w) c -> b c h w', h=H_s, w=H_s)

        return self.template, x_s

    def set_online(self, x_t, x_ot):
        ### conv embeddings for x_t
        x_t = self.patch_embed1(x_t)
        x_t = self.pos_drop(x_t)
        for blk in self.blocks1:
            x_t = blk(x_t)
        x_t = self.patch_embed2(x_t)
        for blk in self.blocks2:
            x_t = blk(x_t)
        x_t = self.patch_embed3(x_t)
        x_t = x_t.flatten(2).permute(0, 2, 1)  # BCHW --> BNC
        x_t = self.patch_embed4(x_t)

        ### conv embeddings for x_ot
        x_ot = self.patch_embed1(x_ot)
        x_ot = self.pos_drop(x_ot)
        for blk in self.blocks1:
            x_ot = blk(x_ot)
        x_ot = self.patch_embed2(x_ot)
        for blk in self.blocks2:
            x_ot = blk(x_ot)
        x_ot = self.patch_embed3(x_ot)
        x_ot = x_ot.flatten(2).permute(0, 2, 1)
        x_ot = self.patch_embed4(x_ot)

        B, C = x_t.size(0), x_t.size(-1)
        H_t = W_t = self.grid_size_t

        x_t = x_t + self.pos_embed_t
        x_ot = x_ot + self.pos_embed_t
        x_ot = x_ot.reshape(1, -1, x_ot.size(-1))  # [1, num_ot * H_t * W_t, C]
        x = torch.cat([x_t, x_ot], dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks3:
            x = blk.set_online(x, H_t, W_t)

        x_t = x[:, :H_t * W_t]
        x_t = rearrange(x_t, 'b (h w) c -> b c h w', h=H_t, w=W_t)

        self.template = x_t


def get_mixformer_convmae(config, train):
    img_size_s = config.DATA.SEARCH.SIZE
    img_size_t = config.DATA.TEMPLATE.SIZE
    if config.MODEL.VIT_TYPE == 'convmae_base':
        vit = ConvViT(
        img_size_s=img_size_s, img_size_t=img_size_t, patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    elif config.MODEL.VIT_TYPE == 'convmae_large':
        vit = ConvViT(
        img_size_s=img_size_s, img_size_t=img_size_t, patch_size=[4, 2, 2], embed_dim=[384, 768, 1024], depth=[2, 2, 20], num_heads=16, mlp_ratio=[4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    else:
        raise KeyError(f"VIT_TYPE shoule set to 'convmae_base' or 'convmae_large'")


    if config.MODEL.BACKBONE.PRETRAINED and train:
        ckpt_path = config.MODEL.BACKBONE.PRETRAINED_PATH
        ckpt = torch.load(ckpt_path, map_location='cpu')['model']
        new_dict = {}
        for k, v in ckpt.items():
            if 'pos_embed' not in k and 'mask_token' not in k:
                new_dict[k] = v
        missing_keys, unexpected_keys = vit.load_state_dict(new_dict, strict=False)
        if is_main_process():
            print("Load pretrained backbone checkpoint from:", ckpt_path)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained ViT done.")

    return vit


class MixFormer(nn.Module):
    def __init__(self, backbone, box_head, head_type="CORNER"):
        """ Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.box_head = box_head
        self.head_type = head_type

    def forward(self, template, online_template, search, run_score_head=False, gt_bboxes=None):
        # search: (b, c, h, w)
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, online_template, search = self.backbone(template, online_template, search)
        # search shape: (b, 384, 20, 20)
        # Forward the corner head
        return self.forward_box_head(search)

    def forward_test(self, search, run_score_head=True, gt_bboxes=None):
        # search: (b, c, h, w)
        if search.dim() == 5:
            search = search.squeeze(0)
        template, search = self.backbone.forward_test(search)
        # search (b, 384, 20, 20)
        # Forward the corner head
        return self.forward_box_head(search)

    def set_online(self, template, online_template):
        if template.dim() == 5:
            template = template.squeeze(0)
        if online_template.dim() == 5:
            online_template = online_template.squeeze(0)
        self.backbone.set_online(template, online_template)

    def forward_box_head(self, search):
        """
        :param search: (b, c, h, w)
        :return:
        """
        if "CORNER" in self.head_type:
            # run the corner head
            b = search.size(0)
            outputs_coord = box_xyxy_to_cxcywh(self.box_head(search))
            outputs_coord_new = outputs_coord.view(b, 1, 4)
            out_dict = {'pred_boxes': outputs_coord_new}
            return out_dict, outputs_coord_new
        else:
            raise KeyError


def build_mixformer_convmae(cfg, train=True) -> MixFormer:
    backbone = get_mixformer_convmae(cfg, train)  # backbone without positional encoding and attention mask
    box_head = build_box_head(cfg)  # a simple corner head
    model = MixFormer(
        backbone,
        box_head,
        head_type=cfg.MODEL.HEAD_TYPE
    )

    return model









'''
Reversible Vision Transformers模型代码
'''
import sys
from functools import partial
import torch
from torch import nn
from torch.autograd import Function as Function

from slowfast.models.attention import MultiScaleAttention, attention_pool
from slowfast.models.common import Mlp, TwoStreamFusion, drop_path
from slowfast.models.utils import round_width


class ReversibleMViT(nn.Module):
    """
    Reversible model builder. This builds the reversible transformer encoder
    and allows reversible training.

    Karttikeya Mangalam, Haoqi Fan, Yanghao Li, Chao-Yuan Wu, Bo Xiong,
    Christoph Feichtenhofer, Jitendra Malik
    "Reversible Vision Transformers"

    https://openaccess.thecvf.com/content/CVPR2022/papers/Mangalam_Reversible_Vision_Transformers_CVPR_2022_paper.pdf
    """

    def __init__(self, config, model):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
            model (nn.Module): parent MViT module this module forms
                a reversible encoder in.
        """

        super().__init__()
        self.cfg = config

        embed_dim = self.cfg.MVIT.EMBED_DIM
        depth = self.cfg.MVIT.DEPTH
        num_heads = self.cfg.MVIT.NUM_HEADS
        mlp_ratio = self.cfg.MVIT.MLP_RATIO
        qkv_bias = self.cfg.MVIT.QKV_BIAS

        drop_path_rate = self.cfg.MVIT.DROPPATH_RATE
        self.dropout = config.MVIT.DROPOUT_RATE
        self.pre_q_fusion = self.cfg.MVIT.REV.PRE_Q_FUSION
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        input_size = model.patch_dims

        self.layers = nn.ModuleList([])
        self.no_custom_backward = False

        if self.cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(self.cfg.MVIT.DIM_MUL)):
            dim_mul[self.cfg.MVIT.DIM_MUL[i][0]] = self.cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(self.cfg.MVIT.HEAD_MUL)):
            head_mul[self.cfg.MVIT.HEAD_MUL[i][0]] = self.cfg.MVIT.HEAD_MUL[i][
                1
            ]

        pool_q = model.pool_q
        pool_kv = model.pool_kv
        stride_q = model.stride_q
        stride_kv = model.stride_kv

        for i in range(depth):

            num_heads = round_width(num_heads, head_mul[i])

            # Upsampling inside the MHPA, input to the Q-pooling block is lower C dimension
            # This localizes the feature changes in a single block, making more computation reversible.
            embed_dim = round_width(
                embed_dim, dim_mul[i - 1] if i > 0 else 1.0, divisor=num_heads
            )
            dim_out = round_width(
                embed_dim,
                dim_mul[i],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )

            if i in self.cfg.MVIT.REV.BUFFER_LAYERS:
                layer_type = StageTransitionBlock
                input_mult = 2 if "concat" in self.pre_q_fusion else 1
            else:
                layer_type = ReversibleBlock
                input_mult = 1

            dimout_correction = (
                2 if (input_mult == 2 and "concat" in self.pre_q_fusion) else 1
            )

            self.layers.append(
                layer_type(
                    dim=embed_dim
                    * input_mult,  # added only for concat fusion before Qpooling layers
                    input_size=input_size,
                    dim_out=dim_out * input_mult // dimout_correction,
                    num_heads=num_heads,
                    cfg=self.cfg,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    kernel_q=pool_q[i] if len(pool_q) > i else [],
                    kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                    stride_q=stride_q[i] if len(stride_q) > i else [],
                    stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                    layer_id=i,
                    pre_q_fusion=self.pre_q_fusion,
                )
            )
            # F is the attention block
            self.layers[-1].F.thw = input_size

            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride
                    for size, stride in zip(input_size, stride_q[i])
                ]

        embed_dim = dim_out

    @staticmethod
    def vanilla_backward(h, layers, buffer):
        """
        Using rev layers without rev backpropagation. Debugging purposes only.
        Activated with self.no_custom_backward.
        """

        # split into hidden states (h) and attention_output (a)
        h, a = torch.chunk(h, 2, dim=-1)
        for _, layer in enumerate(layers):
            a, h = layer(a, h)

        return torch.cat([a, h], dim=-1)

    def forward(self, x):

        # process the layers in a reversible stack and an irreversible stack.
        stack = []
        for l_i in range(len(self.layers)):
            if isinstance(self.layers[l_i], StageTransitionBlock):
                stack.append(("StageTransition", l_i))
            else:
                if len(stack) == 0 or stack[-1][0] == "StageTransition":
                    stack.append(("Reversible", []))
                stack[-1][1].append(l_i)

        for layer_seq in stack:

            if layer_seq[0] == "StageTransition":
                x = self.layers[layer_seq[1]](x)

            else:
                x = torch.cat([x, x], dim=-1)

                # no need for custom backprop in eval/model stat log
                if not self.training or self.no_custom_backward:
                    executing_fn = ReversibleMViT.vanilla_backward
                else:
                    executing_fn = RevBackProp.apply

                x = executing_fn(
                    x,
                    self.layers[layer_seq[1][0] : layer_seq[1][-1] + 1],
                    [],  # buffer activations
                )

        # Apply dropout
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)

        return x


class RevBackProp(Function):
    """
    Custom Backpropagation function to allow (A) flusing memory in foward
    and (B) activation recomputation reversibly in backward for gradient calculation.

    Inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(
        ctx,
        x,
        layers,
        buffer_layers,  # List of layer ids for int activation to buffer
    ):
        """
        Reversible Forward pass. Any intermediate activations from `buffer_layers` are
        cached in ctx for forward pass. This is not necessary for standard usecases.
        Each reversible layer implements its own forward pass logic.
        """
        buffer_layers.sort()

        X_1, X_2 = torch.chunk(x, 2, dim=-1)

        intermediate = []

        for layer in layers:

            X_1, X_2 = layer(X_1, X_2)

            if layer.layer_id in buffer_layers:
                intermediate.extend([X_1.detach(), X_2.detach()])

        if len(buffer_layers) == 0:
            all_tensors = [X_1.detach(), X_2.detach()]
        else:
            intermediate = [torch.LongTensor(buffer_layers), *intermediate]
            all_tensors = [X_1.detach(), X_2.detach(), *intermediate]

        ctx.save_for_backward(*all_tensors)
        ctx.layers = layers

        return torch.cat([X_1, X_2], dim=-1)

    @staticmethod
    def backward(ctx, dx):
        """
        Reversible Backward pass. Any intermediate activations from `buffer_layers` are
        recovered from ctx. Each layer implements its own loic for backward pass (both
        activation recomputation and grad calculation).
        """
        dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)

        # retrieve params from ctx for backward
        X_1, X_2, *int_tensors = ctx.saved_tensors

        # no buffering
        if len(int_tensors) != 0:
            buffer_layers = int_tensors[0].tolist()

        else:
            buffer_layers = []

        layers = ctx.layers

        for _, layer in enumerate(layers[::-1]):

            if layer.layer_id in buffer_layers:

                X_1, X_2, dX_1, dX_2 = layer.backward_pass(
                    Y_1=int_tensors[
                        buffer_layers.index(layer.layer_id) * 2 + 1
                    ],
                    Y_2=int_tensors[
                        buffer_layers.index(layer.layer_id) * 2 + 2
                    ],
                    dY_1=dX_1,
                    dY_2=dX_2,
                )

            else:

                X_1, X_2, dX_1, dX_2 = layer.backward_pass(
                    Y_1=X_1,
                    Y_2=X_2,
                    dY_1=dX_1,
                    dY_2=dX_2,
                )

        dx = torch.cat([dX_1, dX_2], dim=-1)

        del int_tensors
        del dX_1, dX_2, X_1, X_2

        return dx, None, None


class StageTransitionBlock(nn.Module):
    """
    Blocks for changing the feature dimensions in MViT (using Q-pooling).
    See Section 3.3.1 in paper for details.
    """

    def __init__(
        self,
        dim,
        input_size,
        dim_out,
        num_heads,
        mlp_ratio,
        qkv_bias,
        drop_path,
        kernel_q,
        kernel_kv,
        stride_q,
        stride_kv,
        cfg,
        norm_layer=nn.LayerNorm,
        pre_q_fusion=None,
        layer_id=0,
    ):
        """
        Uses the same structure of F and G functions as Reversible Block except
        without using reversible forward (and backward) pass.
        """
        super().__init__()

        self.drop_path_rate = drop_path

        embed_dim = dim

        self.F = AttentionSubBlock(
            dim=embed_dim,
            input_size=input_size,
            num_heads=num_heads,
            cfg=cfg,
            dim_out=dim_out,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
        )

        self.G = MLPSubblock(
            dim=dim_out,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )

        self.layer_id = layer_id

        self.is_proj = False
        self.has_cls_embed = cfg.MVIT.CLS_EMBED_ON

        self.is_conv = False
        self.pool_first = cfg.MVIT.POOL_FIRST
        self.mode = cfg.MVIT.MODE
        self.pre_q_fuse = TwoStreamFusion(pre_q_fusion, dim=dim)

        if cfg.MVIT.REV.RES_PATH == "max":
            self.res_conv = False
            self.pool_skip = nn.MaxPool3d(
                # self.attention.attn.pool_q.kernel_size,
                [s + 1 if s > 1 else s for s in self.F.attn.pool_q.stride],
                self.F.attn.pool_q.stride,
                [int(k // 2) for k in self.F.attn.pool_q.stride],
                # self.attention.attn.pool_q.padding,
                ceil_mode=False,
            )

        elif cfg.MVIT.REV.RES_PATH == "conv":
            self.res_conv = True
        else:
            raise NotImplementedError

        # Add a linear projection in residual branch
        if embed_dim != dim_out:
            self.is_proj = True
            self.res_proj = nn.Linear(embed_dim, dim_out, bias=True)

    def forward(
        self,
        x,
    ):
        """
        Forward logic is similar to MultiScaleBlock with Q-pooling.
        """
        x = self.pre_q_fuse(x)

        # fork tensor for residual connections
        x_res = x

        # This uses conv to pool the residual hidden features
        # but done before pooling only if not pool_first
        if self.is_proj and not self.pool_first:
            x_res = self.res_proj(x_res)

        if self.res_conv:

            # Pooling the hidden features with the same conv as Q
            N, L, C = x_res.shape

            # This handling is the same as that of q in MultiScaleAttention
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.F.attn.num_heads

            # Output is (B, N, L, C)
            x_res = x_res.reshape(N, L, fold_dim, C // fold_dim).permute(
                0, 2, 1, 3
            )

            x_res, _ = attention_pool(
                x_res,
                self.F.attn.pool_q,
                # thw_shape = self.attention.attn.thw,
                thw_shape=self.F.thw,
                has_cls_embed=self.has_cls_embed,
                norm=self.F.attn.norm_q
                if hasattr(self.F.attn, "norm_q")
                else None,
            )
            x_res = x_res.permute(0, 2, 1, 3).reshape(N, x_res.shape[2], C)

        else:
            # Pooling the hidden features with max op
            x_res, _ = attention_pool(
                x_res,
                self.pool_skip,
                thw_shape=self.F.attn.thw,
                has_cls_embed=self.has_cls_embed,
            )

        # If pool_first then project to higher dim now
        if self.is_proj and self.pool_first:
            x_res = self.res_proj(x_res)

        x = self.F(x)
        x = x_res + x
        x = x + self.G(x)

        x = drop_path(x, drop_prob=self.drop_path_rate, training=self.training)

        return x


class ReversibleBlock(nn.Module):
    """
    Reversible Blocks for Reversible Vision Transformer and also
    for state-preserving blocks in Reversible MViT. See Section
    3.3.2 in paper for details.
    """

    def __init__(
        self,
        dim,
        input_size,
        dim_out,
        num_heads,
        mlp_ratio,
        qkv_bias,
        drop_path,
        kernel_q,
        kernel_kv,
        stride_q,
        stride_kv,
        cfg,
        norm_layer=nn.LayerNorm,
        layer_id=0,
        **kwargs
    ):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """
        super().__init__()

        self.drop_path_rate = drop_path

        self.F = AttentionSubBlock(
            dim=dim,
            input_size=input_size,
            num_heads=num_heads,
            cfg=cfg,
            dim_out=dim_out,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
        )

        self.G = MLPSubblock(
            dim=dim,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )

        self.layer_id = layer_id

        self.seeds = {}

    def seed_cuda(self, key):
        """
        Fix seeds to allow for stochastic elements such as
        dropout to be reproduced exactly in activation
        recomputation in the backward pass.
        """

        # randomize seeds
        # use cuda generator if available
        if (
            hasattr(torch.cuda, "default_generators")
            and len(torch.cuda.default_generators) > 0
        ):
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)

        self.seeds[key] = seed
        torch.manual_seed(self.seeds[key])

    def forward(self, X_1, X_2):
        """
        forward pass equations:
        Y_1 = X_1 + Attention(X_2), F = Attention
        Y_2 = X_2 + MLP(Y_1), G = MLP
        """

        self.seed_cuda("attn")
        # Y_1 : attn_output
        f_X_2 = self.F(X_2)

        self.seed_cuda("droppath")
        f_X_2_dropped = drop_path(
            f_X_2, drop_prob=self.drop_path_rate, training=self.training
        )

        # Y_1 = X_1 + f(X_2)
        Y_1 = X_1 + f_X_2_dropped

        # free memory
        del X_1

        self.seed_cuda("FFN")
        g_Y_1 = self.G(Y_1)

        torch.manual_seed(self.seeds["droppath"])
        g_Y_1_dropped = drop_path(
            g_Y_1, drop_prob=self.drop_path_rate, training=self.training
        )

        # Y_2 = X_2 + g(Y_1)
        Y_2 = X_2 + g_Y_1_dropped

        del X_2

        return Y_1, Y_2

    def backward_pass(
        self,
        Y_1,
        Y_2,
        dY_1,
        dY_2,
    ):
        """
        equation for activation recomputation:
        X_2 = Y_2 - G(Y_1), G = MLP
        X_1 = Y_1 - F(X_2), F = Attention
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():

            Y_1.requires_grad = True

            torch.manual_seed(self.seeds["FFN"])
            g_Y_1 = self.G(Y_1)

            torch.manual_seed(self.seeds["droppath"])
            g_Y_1 = drop_path(
                g_Y_1, drop_prob=self.drop_path_rate, training=self.training
            )

            g_Y_1.backward(dY_2, retain_graph=True)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass.
        with torch.no_grad():

            X_2 = Y_2 - g_Y_1
            del g_Y_1

            dY_1 = dY_1 + Y_1.grad
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad = True

            torch.manual_seed(self.seeds["attn"])
            f_X_2 = self.F(X_2)

            torch.manual_seed(self.seeds["droppath"])
            f_X_2 = drop_path(
                f_X_2, drop_prob=self.drop_path_rate, training=self.training
            )

            f_X_2.backward(dY_1, retain_graph=True)

        # propagate reverse computed acitvations at the start of
        # the previou block for backprop.s
        with torch.no_grad():

            X_1 = Y_1 - f_X_2

            del f_X_2, Y_1
            dY_2 = dY_2 + X_2.grad

            X_2.grad = None
            X_2 = X_2.detach()

        return X_1, X_2, dY_1, dY_2


class MLPSubblock(nn.Module):
    """
    This creates the function G such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim,
        mlp_ratio,
        norm_layer=nn.LayerNorm,
    ):

        super().__init__()
        self.norm = norm_layer(dim, eps=1e-6, elementwise_affine=True)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
        )

    def forward(self, x):
        return self.mlp(self.norm(x))


class AttentionSubBlock(nn.Module):
    """
    This creates the function F such that the entire block can be
    expressed as F(G(X)). Includes pre-LayerNorm.
    """

    def __init__(
        self,
        dim,
        input_size,
        num_heads,
        cfg,
        dim_out=None,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
    ):

        super().__init__()
        self.norm = norm_layer(dim, eps=1e-6, elementwise_affine=True)

        # This will be set externally during init
        self.thw = None

        # the actual attention details are the same as Multiscale
        # attention for MViTv2 (with channel up=projection inside block)
        # can also implement no upprojection attention for vanilla ViT
        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            input_size=input_size,
            num_heads=num_heads,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            drop_rate=cfg.MVIT.DROPOUT_RATE,
            qkv_bias=cfg.MVIT.QKV_BIAS,
            has_cls_embed=cfg.MVIT.CLS_EMBED_ON,
            mode=cfg.MVIT.MODE,
            pool_first=cfg.MVIT.POOL_FIRST,
            rel_pos_spatial=cfg.MVIT.REL_POS_SPATIAL,
            rel_pos_temporal=cfg.MVIT.REL_POS_TEMPORAL,
            rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
            residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
            separate_qkv=cfg.MVIT.SEPARATE_QKV,
        )

    def forward(self, x):
        out, _ = self.attn(self.norm(x), self.thw)
        return out
