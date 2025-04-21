from typing import Literal
from functools import partial
import torch
from torch import nn
from timm.models.vision_transformer import (
    LayerType,
    Block,
    Mlp,
    PatchEmbed,
    VisionTransformer as TimmViT,
    checkpoint_filter_fn,
    build_model_with_cfg,
)
from .._base import ModelBase


class VisionTransformer(TimmViT, ModelBase):
    def __init__(
            self,
            img_size: int|tuple[int, int] = 224,
            patch_size: int|tuple[int, int] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'avgmax', 'max', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_bias: bool = True,
            init_values: float|None = None,
            class_token: bool = True,
            pos_embed: str = 'learn',
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            final_norm: bool = True,
            fc_norm: bool|None = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            fix_init: bool = False,
            embed_layer: function = PatchEmbed,
            embed_norm_layer: LayerType|None = None,
            norm_layer: LayerType|None = None,
            act_layer: LayerType|None = None,
            block_fn: type[nn.Module] = Block,
            mlp_layer: type[nn.Module] = Mlp,
    ) -> None:
        super(VisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            init_values=init_values,
            class_token=class_token,
            pos_embed=pos_embed,
            no_embed_class=no_embed_class,
            reg_tokens=reg_tokens,
            pre_norm=pre_norm,
            final_norm=final_norm,
            fc_norm=fc_norm,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            drop_rate=drop_rate,
            pos_drop_rate=pos_drop_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            fix_init=fix_init,
            embed_layer=embed_layer,
            embed_norm_layer=embed_norm_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
            mlp_layer=mlp_layer,
        )
    
    def get_arch(self) -> Literal['cnn', 'transformer']:
        return 'transformer'

    def forward_stem(self, x: torch.Tensor):
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        return x

    def get_layers(self):
        return self.blocks

    def forward_pool(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x

    def get_head(self):
        return self.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_stem(x)
        feats = {
            'feats': [],
            'preact_feats': [],
            'pooled_feat': None,
        }
        for block in self.blocks:
            x = block.forward(x)
            feats['preact_feats'].append(x)
            feats['feats'].append(x)
        x = self.forward_pool(x)
        feats['pooled_feat'] = x
        x = self.head(x)
        return x, feats


def _create_vision_transformer(variant: str, pretrained: bool = False, **kwargs) -> VisionTransformer:
    out_indices = kwargs.pop('out_indices', 3)
    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = kwargs.pop('pretrained_strict', True)
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        feature_cfg=dict(out_indices=out_indices, feature_cls='getter'),
        **kwargs,
    )


def vit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

def vit_small_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

def vit_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

def vit_large_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model
