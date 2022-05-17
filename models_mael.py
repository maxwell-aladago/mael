# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
# from collections import OrderedDict
import torch
import torch.nn as nn

# from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from transformers import BertModel
import torch.nn.functional as F
from util.misc import all_gather_batch_with_grad, get_world_size, get_rank
from models_mae import MaskedAutoencoderViT
from bertmodel import ClsBertModel
from vitl import ViTL


class MaskedAutoencoderWithLanguage(MaskedAutoencoderViT):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, text_encoder, num_text_tokens=8, text_output='cls_token',
                 img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):

        super().__init__(img_size, patch_size, in_chans,
                         embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_pix_loss)

        num_patches = self.patch_embed.num_patches
        # --------------------------------------------------------------------------
        # Text Encoder Specifics
        self.text_encoder = text_encoder
        text_encoder_embed_dim = self.text_encoder.embed_dim

        self.text_repr = nn.Linear(text_encoder_embed_dim, decoder_embed_dim, bias=True)
        self.text_norm = norm_layer(text_encoder_embed_dim)

        self.text_output = text_output
        self.num_text_tokens = num_text_tokens if self.text_output == "all_tokens" else 1

        # ---------------------------------------------------------------------------
        # contrastive learning params
        # self.tau = 0.1
        # self.labels = None
        # self.masks = None
        # self.last_local_batch_size = None
        # --------------------------------------------------------------------------

        # class tokens projections
        # in_dim = self.decoder_embed
        # mlp_dim = in_dim * 2
        # out_dim = 512

        # self.out_project = nn.Sequential(OrderedDict([
        #     ("layer1", nn.Linear(in_dim, mlp_dim)),
        #     ("bn1", nn.SyncBatchNorm(mlp_dim)),
        #     ("relu1", nn.ReLU(inplace=True)),
        #     ("layer2", nn.Linear(mlp_dim, mlp_dim)),
        #     ("bn2", nn.SyncBatchNorm(mlp_dim)),
        #     ("relu2", nn.ReLU(inplace=True)),
        #     ("layer3", nn.Linear(mlp_dim, out_dim)),
        # ]))

        # update decoder pos_embed
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + self.num_text_tokens,
                                                          decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self._initialize_weights()

    def _initialize_weights(self):

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5),
                                                    cls_token=True, text_tokens=self.num_text_tokens)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        for n, m in self.text_repr.named_modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward_decoder(self, x, ids_restore, text_tks=None):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        N, L, d = x_.shape
        x = torch.cat([x[:, :1, :], x_, text_tks], dim=1)  # append cls and text tokens

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls and text tokens
        cls_tkn = x[:, 0, :]
        text_tkn = x[:, L + 1, :]
        x = x[:, 1:L + 1, :]
        return x, cls_tkn, text_tkn

    def forward_text_encoder(self, captions):
        x = self.text_encoder(**captions)

        x = self.text_norm(x)

        # use just the cls-token or an average of all tokens or all the text tokens
        if self.text_output == "all_tokens":
            x = self.text_repr(x)
        elif self.text_output == "avg":
            x = self.text_repr(x.mean(dim=1, keepdims=True))
        else:
            x = self.text_repr(x[:, 0]).unsqueeze(dim=1)

        return x

    # def forward_contrastive_loss(self, img_rep, text_rep):
    #     img_rep = self.out_project(img_rep)
    #     text_rep = self.out_project(text_rep)
    #
    #     q_a = F.normalize(img_rep, dim=-1, p=2)
    #     q_b = F.normalize(text_rep, dim=-1, p=2)
    #
    #     local_batch_size = q_a.size(0)
    #
    #     k_a, k_b = all_gather_batch_with_grad([q_a, q_b])
    #
    #     if local_batch_size != self.last_local_batch_size:
    #         self.labels = local_batch_size * get_rank() + torch.arange(
    #             local_batch_size, device=q_a.device
    #         )
    #         total_batch_size = local_batch_size * get_world_size()
    #         self.masks = F.one_hot(self.labels, total_batch_size) * 1e9
    #         self.last_local_batch_size = local_batch_size
    #
    #     logits_aa = torch.matmul(q_a, k_a.transpose(0, 1)) / self.tau
    #     logits_aa = logits_aa - self.masks
    #     logits_bb = torch.matmul(q_b, k_b.transpose(0, 1)) / self.tau
    #     logits_bb = logits_bb - self.masks
    #     logits_ab = torch.matmul(q_a, k_b.transpose(0, 1)) / self.tau
    #     logits_ba = torch.matmul(q_b, k_a.transpose(0, 1)) / self.tau
    #
    #     loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), self.labels)
    #     loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), self.labels)
    #     loss = (loss_a + loss_b) / 2  # divide by 2 to average over all samples
    #
    #     return loss

    def forward(self, imgs, mask_ratio=0.75, captions=None):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        text_tkns = self.forward_text_encoder(captions)

        # move decoder embed and text embed here
        pred, cls_tkn, text_tkn = self.forward_decoder(latent, ids_restore, text_tkns)  # [N, L, p*p*3]
        mae_loss = self.forward_loss(imgs, pred, mask)

        # get the cls_tokens from text and images
        # put them through a projection and contrast them.
        # contrastive_loss = self.forward_contrastive_loss(cls_tkn, text_tkn)

        # total_loss = (mae_loss + contrastive_loss) / 2.0

        return mae_loss, pred, mask


def language_model(name="google/bert_uncased_L-8_H-256_A-4", sq_ln=16, pretrained=True):

    if 'bert' in name:
        name_ = "google/bert_uncased_L-10_H-256_A-4"
        model = ClsBertModel.from_pretrained(name_)
        model = ClsBertModel(model.config)
        state_dict = BertModel.from_pretrained(name).state_dict()

        msg = model.load_state_dict(state_dict, strict=False)

        del state_dict
        # print(msg.missing_keys)
        # if not pretrained:
        #     model = ClsBertModel(model.config)
        # else:
        #     for n, p in model.named_parameters():
        #         if 'cls_token' not in name:
        #             p.requires_grad = False
        # print(model)
        # exit()
    else:
        model_kwargs = dict(embed_dim=384, depth=12, num_heads=8, mlp_ratio=4,
                            norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0)

        model = ViTL(vocab_size=30522, sq_ln=sq_ln, **model_kwargs)

    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    text_encoder = language_model(
        name=kwargs.pop('l_model'),
        sq_ln=kwargs.get('num_text_tokens'),
        pretrained=kwargs.pop('pretrained'))

    model = MaskedAutoencoderWithLanguage(
        text_encoder=text_encoder,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderWithLanguage(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderWithLanguage(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
