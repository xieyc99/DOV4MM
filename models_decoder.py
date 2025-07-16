from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from utils.pos_embed import get_2d_sincos_pos_embed
import os
from utils.util import *
import encoder_model.vit4mae as vit4mae
    
class MaskedAutoencoderViT2(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, need_encoder=True, encoder_path=None, encoder_model='vit_base', encoder_method=None, align_feature=False):
        super().__init__()

        # --------------------------------------------------------------------------
        self.num_patches = int((img_size/patch_size)**2)
        self.patch_size = patch_size
        print('num_patches:', self.num_patches)
        self.align_feature = align_feature
        self.encoder_model = encoder_model

        # --------------------------------------------------------------------------

        self.encoder, embed_dim = build_model(encoder_model, encoder_method)
        if need_encoder:
            self.encoder = load_model(self.encoder, encoder_path)
        
        # --------------------------------------------------------------------------

        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if not align_feature:
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        else:
            self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def delete_masked_patches(self, a, b_mask):
        batch_size, num_patches, dim = a.shape
        _, num_mask = b_mask.shape

        mask = torch.ones(batch_size, num_patches, device=a.device)

        batch_indices = torch.arange(batch_size, device=a.device).unsqueeze(1).expand(-1, num_mask)

        mask[batch_indices, b_mask] = 0

        mask = mask.unsqueeze(-1)  # 形状: (batch_size, num_patches, 1)

        a_masked = a * mask

        return a_masked

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # print('ids_shuffle:', ids_shuffle.shape)
        # print('ids_shuffle:', ids_shuffle[:, :])
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # print('ids_restore:', ids_restore.shape)
        # print('ids_restore:', ids_restore[:, :])

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_not_keep = ids_shuffle[:, len_keep:]
        # print('ids_keep:', ids_keep)
        # print('ids_not_keep:', ids_not_keep)
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_masked = x.clone()
        x_masked = self.delete_masked_patches(x_masked, ids_not_keep)
        # print('x_masked:', x_masked.shape)
        # print(x_masked[0, 73, :])

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # print('mask:', mask)
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # print('mask:', mask.shape)
        # print('mask:', mask)

        return x_masked, mask, ids_restore, ids_keep

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        # print('---------decoder---------')
        # print('x:', x.shape)
        x = self.decoder_embed(x)
        # print('x:', x.shape)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        # print('mask_tokens:', mask_tokens.shape)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        # print('x_:', x_.shape)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # print('x_:', x_.shape)
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # print('---------loss---------')
        # print('imgs:', imgs.shape)
        # print('pred:', pred.shape)
        # print('mask:', mask.shape)
        if self.align_feature:
            target = self.encoder.forward_features(imgs)
            if 'swin' not in self.encoder_model and 'mixmim' not in self.encoder_model:
                target = target[:, 1:]
            # print('target:', target.shape)
        else:
            target = self.patchify(imgs)
        # print('target:', target.shape)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        # print('mask.sum():', mask.sum())
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    # def forward(self, imgs, mask_ratio=0.75):
    #     latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
    #     pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
    #     loss = self.forward_loss(imgs, pred, mask)
    #     return loss, pred, mask
    
    def forward(self, imgs, mask_ratio=0.75):
        with torch.no_grad():
            x = self.patchify(imgs)
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
            x = self.unpatchify(x)
            # print('x:', x.shape)
            # print('ids_restore:', ids_restore.shape)
            # print('ids_keep:', ids_keep.shape)

            x = self.encoder.forward_features(x)
            # print('x:', x.shape)

        if 'swin' in self.encoder_model or 'mixmim' in self.encoder_model:
            N, L, D = x.shape
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        else:
            x_ = x[:, 1:]  # delete class token
            N, L, D = x_.shape
            x_masked = torch.gather(x_, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # delete masked token
            x_masked = torch.cat((x[:, :1], x_masked), dim=1)
            # print('x_masked:', x_masked.shape)


        pred = self.forward_decoder(x_masked, ids_restore)
        # print('pred:', pred.shape)

        loss = self.forward_loss(imgs, pred, mask)
        # print('loss:', loss.shape)

        return x, loss
    

if __name__=='__main__':
    set_seed(99)
    # decoder = MaskedDecoderViT(embed_dim=768).cuda()

    # x = torch.rand((32, 197, 768)).cuda()
    # target = torch.rand((32, 196, 768)).cuda()
    # root = r'D:\Exp\DOV4MIM\data\ImageNet\ImageNet-20000-75-80-5'
    # mask_pos = torch.load(os.path.join(root, 'mask_poses_train_1.pth'), map_location='cuda')
    # mask_pos = mask_pos[:32]
    # print('mask_pos:', mask_pos.shape)

    # output, loss = decoder(x, target, mask_pos)

    img = torch.rand((2, 3, 224, 224))
    model = MaskedAutoencoderViT2(embed_dim=768, encoder_path='E:\ckpt\DOV4CL\Experiments\imagenet\mae_pretrain_vit_base.pth')
    x = model(img)