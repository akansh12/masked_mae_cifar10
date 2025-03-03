import torch
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from timm.models.vision_transformer import Block, PatchEmbed


class MAE_Encoder(torch.nn.Module):
    def __init__(
            self,
            img_size = 32,
            patch_size = 2,
            in_chans = 3,
            emb_dim = 192,
            num_layers = 12,
            num_heads = 3,
            mask_ratio = 0.75,
            mlp_dim = 768
            ) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding  = nn.Parameter(torch.empty(1, (img_size // patch_size) ** 2 + 1, emb_dim))

        self.patchify = PatchEmbed(
            img_size = img_size,
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = emb_dim
        )
        ### Encoder model
        self.encoder = nn.ModuleList([
            Block(
                dim = emb_dim,
                num_heads = num_heads,
                mlp_ratio = mlp_dim / emb_dim,
                qkv_bias = True,
                norm_layer = nn.LayerNorm,
            ) for _ in range(num_layers)
        ])

        self.norm_layer = nn.LayerNorm(emb_dim)

        self.initialize_weights()
    
    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embedding, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT: #Code taken from FAIR
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward(self, x, mask_ratio = 0.75):
        x = self.patchify(x)

        #Add position embedding w/o cls token
        x = x + self.pos_embedding[:, 1:, :]

        #masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        #Append cls token
        cls_token = self.cls_token + self.pos_embedding[:, 0:1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1) #Expand cls token to all batches
        x = torch.cat((cls_token, x), dim=1)

        for block in self.encoder:
            x = block(x)

        x = self.norm_layer(x)

        return x, mask, ids_restore

class MAE_Decoder(torch.nn.Module):
    def __init__(
        self, 
        image_size = 32,
        patch_size = 2,
        emb_dim = 192,
        num_layers = 4,
        num_heads = 3,
        out_chans = 3,
        mlp_dim = 768
    ) -> None:

        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.decoder_pos_embedding = nn.Parameter(torch.empty(1, (image_size // patch_size) ** 2 + 1, emb_dim))

        # self.decoder_emb = nn.Linear(encoder_emb_dim, decoder_embed_dim, bias=True)
        self.decoder = nn.ModuleList([
            Block(
                dim = emb_dim,
                num_heads = num_heads,
                mlp_ratio = mlp_dim / emb_dim,
                qkv_bias = True,
                norm_layer = nn.LayerNorm,
            ) for _ in range(num_layers)
        ])


        self.decoder_norm = nn.LayerNorm(emb_dim)
        self.decoder_pred = nn.Linear(emb_dim, patch_size **2 * out_chans, bias=True)
        self.patch2img = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size // patch_size, w=image_size // patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embedding, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT: #Code taken from FAIR
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        #add position embedding
        x = x + self.decoder_pos_embedding

        for block in self.decoder:
            x = block(x)

        x = self.decoder_norm(x)

        x = self.decoder_pred(x)

        #remove cls token
        x = x[:, 1:, :]

        return x
    
class MAE(torch.nn.Module):
    def __init__(
        self,
        img_size = 32,
        patch_size = 2,
        in_chans = 3,
        encoder_emb_dim = 192,
        encoder_layers = 12,
        encoder_heads = 3,
        encoder_mlp_dim = 768,
        decoder_layers = 4,
        decoder_heads = 3,
        decoder_mlp_dim = 768,
        out_chans = 3
    ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(
            img_size = img_size,
            patch_size = patch_size,
            in_chans = in_chans,
            emb_dim = encoder_emb_dim,
            num_layers = encoder_layers,
            num_heads = encoder_heads,
            mlp_dim = encoder_mlp_dim
        )

        self.decoder = MAE_Decoder(
            image_size = img_size,
            patch_size = patch_size,
            emb_dim = encoder_emb_dim,
            num_layers = decoder_layers,
            num_heads = decoder_heads,
            mlp_dim = decoder_mlp_dim,
            out_chans = out_chans
        )

    def forward(self, x):
        x, mask, ids_restore = self.encoder(x)
        x = self.decoder(x, ids_restore)
        return x, mask

class VIT_Classifier(torch.nn.Module):
    def __init__(
            self, 
            img_size = 32,
            patch_size = 2,
            in_chans = 3,
            num_classes = 10,
            emb_dim = 192,
            num_layers = 12,
            num_heads = 3,
            mlp_dim = 768,
            pretrained = False,
            pretrained_path = None
        ) -> None:
        super().__init__()


        self.encoder_model = MAE_Encoder(
            img_size = img_size,
            patch_size = patch_size,
            in_chans = in_chans,
            emb_dim = emb_dim,
            num_layers = num_layers,
            num_heads = num_heads,
            mlp_dim = mlp_dim
        )
        self.cls_token = self.encoder_model.cls_token
        self.pos_embedding = self.encoder_model.pos_embedding
        self.patchify = self.encoder_model.patchify
        self.encoder = self.encoder_model.encoder
        self.norm_layer = self.encoder_model.norm_layer

        self.classifier = nn.Linear(emb_dim, num_classes)
        if pretrained:
            self.load_state_dict(torch.load(pretrained_path))

    def forward(self, x):
        x = self.patchify(x)

        #Append cls token

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        #Add position embedding
        x = x + self.pos_embedding

        for block in self.encoder:
            x = block(x)

        x = self.norm_layer(x)

        x = x[:, 0, :] #cls token
        x = self.classifier(x)

        return x
        