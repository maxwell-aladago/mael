import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer


class ViTL(VisionTransformer):
    def __init__(self,
                 vocab_size: int,
                 text_sq_ln: int,
                 **kwargs,
                 ):
        super(ViTL, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.text_sq_ln = text_sq_ln

        # initialize all parameters here
        self.init_weights()

        self.patch_embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.text_sq_ln + 1, self.embed_dim))

        nn.init.normal_(self.text_cls_token, std=.02)
        nn.init.normal_(self.text_pos_embed, std=.02)
        nn.init.normal_(self.patch_embed.weight, std=0.02)

    def forward(self, text):
        x = self.patch_embed(text)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.text_pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        return x
