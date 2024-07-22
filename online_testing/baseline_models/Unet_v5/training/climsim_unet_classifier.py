import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import modulus
import nvtx
from layers import (
    Conv1d,
    GroupNorm,
    Linear,
    UNetBlock,
    UNetBlock_noatten,
    UNetBlock_atten,
    ScriptableAttentionOp,
)
from torch.nn.functional import silu
from typing import List

"""
Contains the code for the Unet and its training.
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class ClimsimUnetMetaData_class(modulus.ModelMetaData):
    name: str = "ClimsimUnet_class"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = False
    amp_gpu: bool = False

class ClimsimUnet_class(modulus.Module):
    def __init__(
            self, 
            num_vars_profile: int,
            num_vars_scalar: int, 
            num_vars_profile_out: int,
            num_vars_scalar_out: int, 
            seq_resolution: int = 64,
            label_dim: int = 0,
            augment_dim: int = 0,
            model_channels: int = 128,
            channel_mult: List[int] = [1, 2, 2, 2],
            channel_mult_emb: int = 4,
            num_blocks: int = 4,
            attn_resolutions: List[int] = [16],
            dropout: float = 0.10,
            label_dropout: float = 0.0,
            embedding_type: str = "positional",
            channel_mult_noise: int = 1,
            encoder_type: str = "standard",
            decoder_type: str = "standard",
            resample_filter: List[int] = [1, 1],
            n_model_levels: int = 60,
            # qinput_prune=False, 
            output_prune=False, 
            strato_lev_out=12,
            loc_embedding: bool = False,
            skip_conv: bool = False,
            prev_2d: bool = False,
            skip_phys_tend = False,
            ):
        
        super().__init__(meta=ClimsimUnetMetaData_class())
        # check if hidden_dims is a list of hidden_dims
        self.num_vars_profile = num_vars_profile
        self.num_vars_scalar = num_vars_scalar
        self.num_vars_profile_vars = num_vars_profile_out
        self.num_vars_profile_out = num_vars_profile_out*3
        self.num_vars_scalar_out = num_vars_scalar_out
        self.model_channels = model_channels

        self.in_channels = num_vars_profile + num_vars_scalar + 7 # +(8-1)=7 for the location embedding
        self.out_channels = self.num_vars_profile_out + num_vars_scalar_out

        if num_vars_scalar_out>0:
            raise ValueError('num_vars_scalar_out should be 0')
        if num_vars_profile_out != 1:
            raise ValueError('num_vars_profile_out should be 1')
        # print('1: out_channels', self.out_channels)

        # valid_encoder_types = ["standard", "skip", "residual"]
        valid_encoder_types = ["standard"]
        if encoder_type not in valid_encoder_types:
            raise ValueError(
                f"Invalid encoder_type: {encoder_type}. Must be one of {valid_encoder_types}."
            )

        # valid_decoder_types = ["standard", "skip"]
        valid_decoder_types = ["standard"]
        if decoder_type not in valid_decoder_types:
            raise ValueError(
                f"Invalid decoder_type: {decoder_type}. Must be one of {valid_decoder_types}."
            )

        self.label_dropout = label_dropout
        self.embedding_type = embedding_type

        self.seq_resolution = seq_resolution
        self.label_dim = label_dim
        self.augment_dim = augment_dim
        self.model_channels = model_channels
        self.channel_mult = channel_mult
        self.channel_mult_emb = channel_mult_emb
        self.num_blocks = num_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.channel_mult_noise = channel_mult_noise
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.resample_filter = resample_filter
        self.n_model_levels = n_model_levels
        self.input_padding = (seq_resolution-n_model_levels,0)
        # self.qinput_prune=qinput_prune
        self.output_prune=output_prune
        self.strato_lev_out=strato_lev_out
        self.loc_embedding = loc_embedding
        self.skip_conv = skip_conv
        self.prev_2d = prev_2d
        self.skip_phys_tend = skip_phys_tend

        # emb_channels = model_channels * channel_mult_emb
        # self.emb_channels = emb_channels
        # noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=0.2**0.5)
        block_kwargs = dict(
            # emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=0.5**0.5,
            eps=1e-6,
            resample_filter=resample_filter,
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )


        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = self.in_channels
        caux = self.in_channels
        for level, mult in enumerate(channel_mult):
            res = seq_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                # comment out the first conv layer that supposed to be the input embedding
                # because we will have the input embedding manusally for profile vars and scalar vars
                self.enc[f"{res}_conv"] = Conv1d(
                    in_channels=cin, out_channels=cout, kernel=3, **init
                )
            else:
                self.enc[f"{res}_down"] = UNetBlock_noatten(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
                if encoder_type == "skip":
                    self.enc[f"{res}_aux_down"] = Conv1d(
                        in_channels=caux,
                        out_channels=caux,
                        kernel=0,
                        down=True,
                        resample_filter=resample_filter,
                    )
                    self.enc[f"{res}_aux_skip"] = Conv1d(
                        in_channels=caux, out_channels=cout, kernel=1, **init
                    )
                if encoder_type == "residual":
                    self.enc[f"{res}_aux_residual"] = Conv1d(
                        in_channels=caux,
                        out_channels=cout,
                        kernel=3,
                        down=True,
                        resample_filter=resample_filter,
                        fused_resample=True,
                        **init,
                    )
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = res in attn_resolutions
                if attn:
                    self.enc[f"{res}_block{idx}"] = UNetBlock_atten(
                        in_channels=cin, 
                        out_channels=cout, 
                        emb_channels=0,
                        up=False,
                        down=False,
                        channels_per_head=64,
                        **block_kwargs
                    )
                else:
                    self.enc[f"{res}_block{idx}"] = UNetBlock_noatten(
                        in_channels=cin, 
                        out_channels=cout, 
                        attention=attn,
                        emb_channels=0,
                        up=False,
                        down=False,
                        channels_per_head=64,
                        **block_kwargs
                    )
        skips = [
            block.out_channels for name, block in self.enc.items() if "aux" not in name
        ]

        self.skip_conv_layer = [] #torch.nn.ModuleList()
        # for each skip connection, add a 1x1 conv layer initialized as identity connection, with an option to train the weight
        for idx, skip in enumerate(skips):
            conv = Conv1d(in_channels=skip, out_channels=skip, kernel=1)
            torch.nn.init.dirac_(conv.weight)
            torch.nn.init.zeros_(conv.bias)
            if not self.skip_conv:
                conv.weight.requires_grad = False
                conv.bias.requires_grad = False
            self.skip_conv_layer.append(conv)
        self.skip_conv_layer = torch.nn.ModuleList(self.skip_conv_layer)
            # XX doulbe check if the above is correct

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        self.dec_aux_norm = torch.nn.ModuleDict()
        self.dec_aux_conv = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = seq_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f"{res}_in0"] = UNetBlock_atten(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs
                )
                self.dec[f"{res}_in1"] = UNetBlock_noatten(
                    in_channels=cout, out_channels=cout, **block_kwargs
                )
            else:
                self.dec[f"{res}_up"] = UNetBlock_noatten(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = idx == num_blocks and res in attn_resolutions
                if attn:
                    self.dec[f"{res}_block{idx}"] = UNetBlock_atten(
                        in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                    )
                else:
                    self.dec[f"{res}_block{idx}"] = UNetBlock_noatten(
                        in_channels=cin, out_channels=cout, attention=attn, **block_kwargs
                    )
            if decoder_type == "skip" or level == 0:
                # if decoder_type == "skip" and level < len(channel_mult) - 1:
                #     self.dec[f"{res}_aux_up"] = Conv1d(
                #         in_channels=out_channels,
                #         out_channels=out_channels,
                #         kernel=0,
                #         up=True,
                #         resample_filter=resample_filter,
                #     )
                self.dec_aux_norm[f"{res}_aux_norm"] = GroupNorm(
                    num_channels=cout, eps=1e-6
                )
                ## comment out the last conv layer that supposed to recover the output channels
                ## we will manually recover the output channels
                self.dec_aux_conv[f"{res}_aux_conv"] = Conv1d(
                    in_channels=cout, out_channels=self.out_channels, kernel=3, **init_zero
                )

        # create a 385x8 trainable weight embedding for the input
        self.emb_loc = torch.nn.Parameter(torch.randn(385, 8), requires_grad=True)
                       
    def forward(self, x):
        '''
        x: (batch, num_vars_profile*levels+num_vars_scalar)
        # x_profile: (batch, num_vars_profile, levels)
        # x_scalar: (batch, num_vars_scalar)
        '''

        # if self.qinput_prune:
        #     x = x.clone()  # Clone the tensor to ensure you're not modifying the original tensor in-place
        #     x[:, 60:60+self.strato_lev] = x[:, 60:60+self.strato_lev].clone().zero_()  # Set stratosphere q1 to 0
        #     x[:, 120:120+self.strato_lev] = x[:, 120:120+self.strato_lev].clone().zero_()  # Set stratosphere q2 to 0
        #     x[:, 180:180+self.strato_lev] = x[:, 180:180+self.strato_lev].clone().zero_()  # Set stratosphere q3 to 0

        if not self.prev_2d:
            x = x.clone()
            x[:,-8:-3] = x[:,-8:-3].clone().zero_()

        # split x into x_profile and x_scalar
        x_profile = x[:,:self.num_vars_profile*self.n_model_levels]
        x_scalar = x[:,self.num_vars_profile*self.n_model_levels:-1]
        x_loc = x[:,-1] # location index

        # right now x_loc is only 1-384, use 0 to represent not using position embedding
        if not self.loc_embedding:
            x_loc[:] = 0.0*x_loc[:]
        #convert x_loc to embedding, first use one-hot encoding to convert x_loc to (batch, 385)
        # convert x_loc to one-hot encoding
        x_loc = torch.nn.functional.one_hot(x_loc.to(torch.int64), num_classes=385)
        # convert x_loc from int to float
        x_loc = x_loc.to(torch.float32)
        # convert x_loc to embedding
        x_loc = torch.matmul(x_loc, self.emb_loc) # (batch, 8)

        # print(x_profile.shape, x_scalar.shape, x_loc.shape)

        # reshape x_profile to (batch, num_vars_profile, levels)
        x_profile = x_profile.reshape(-1, self.num_vars_profile, self.n_model_levels)
        # broadcast x_scalar to (batch, num_vars_scalar, levels)
        x_scalar = x_scalar.unsqueeze(2).expand(-1, -1, self.n_model_levels)

        #concatenate x_profile, x_scalar, x_loc to (batch, num_vars_profile+num_vars_scalar+8, levels)
        x = torch.cat((x_profile, x_scalar, x_loc.unsqueeze(2).expand(-1, -1, self.n_model_levels)), dim=1)
        # print('2:', x.shape)
        # x = torch.cat((x_profile, x_scalar), dim=1)
        
        x = torch.nn.functional.pad(x, self.input_padding, "constant", 0.0)
        # print('3:', x.shape)
        # pass the concatenated tensor through the Unet

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if "aux_down" in name:
                aux = block(aux)
            elif "aux_skip" in name:
                x = skips[-1] = x + block(aux)
            elif "aux_residual" in name:
                x = skips[-1] = aux = (x + block(aux)) / 2**0.5
            else:
                # x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                x = block(x)
                skips.append(x)

        new_skips = []
        # for x_tmp, conv_tmp in zip(skips, self.skip_conv_layer):
        #     x_tmp = conv_tmp(x_tmp)
        #     new_skips.append(x_tmp)
        for idx, conv_tmp in enumerate(self.skip_conv_layer):
            x_tmp = conv_tmp(skips[idx])
            new_skips.append(x_tmp)

        aux = None
        tmp = None
        for name, block in self.dec.items():
#             print(name)
            # if "aux" not in name:
            if x.shape[1] != block.in_channels:
                # skip_ind = len(skips) - 1
                # skip_conv = self.skip_conv_layer[skip_ind]
                x = torch.cat([x, new_skips.pop()], dim=1)
            # x = block(x, emb)
            x = block(x)
            # else:
            #     # if "aux_up" in name:
            #     #     aux = block(aux)
            #     if "aux_conv" in name:
            #         tmp = block(silu(tmp))
            #         aux = tmp if aux is None else tmp + aux
            #     elif "aux_norm" in name:
            #         tmp = block(x)
        for name, block in self.dec_aux_norm.items():
            tmp = block(x)
        for name, block in self.dec_aux_conv.items():
            tmp = block(silu(tmp))
            aux = tmp if aux is None else tmp + aux

        # here x should be (batch, output_channels, seq_resolution)
        # remember that self.input_padding = (seq_resolution-n_model_levels,0)
        x = aux
        # print('7:', x.shape)
        if self.input_padding[1]==0:
            y_profile = x[:,:self.num_vars_profile_out,self.input_padding[0]:]
            # y_scalar = x[:,self.num_vars_profile_out:,self.input_padding[0]:]
        else:
            y_profile = x[:,:self.num_vars_profile_out,self.input_padding[0]:-self.input_padding[1]]
            # y_scalar = x[:,self.num_vars_profile_out:,self.input_padding[0]:-self.input_padding[1]]

        #here y_profile is (batch, num_vars_profile_out, levels)
        # reshape y_profile to (batch, num_vars_profile_var, 3, levels)
        # print('8:', y_profile.shape)
        y_profile = y_profile.reshape(-1, self.num_vars_profile_out//3, 3, self.n_model_levels)
        #move 3 to the last dim
        # print('9:', y_profile.shape)
        y_profile = y_profile.permute(0, 1, 3, 2) # here y_profile is (batch, num_vars_profile_var, levels, 3)
        # print('10:', y_profile.shape)
        #reshape y_profile to (batch, num_vars_profile_var*levels, 3)
        y_profile = y_profile.reshape(-1, self.num_vars_profile_vars*self.n_model_levels, 3)
        # softmax along the last dim
        # y_profile = torch.nn.functional.softmax(y_profile, dim=2)
        if self.output_prune:
            y_profile = y_profile.clone()
            # set y_profile[:,0:self.strato_lev,0] to very large:
            y_profile[:,0:self.strato_lev_out,0] = y_profile[:,0:self.strato_lev_out,0].clone().fill_(1e2)
            y_profile[:,0:self.strato_lev_out,1] = y_profile[:,0:self.strato_lev_out,0].clone().zero_()
            y_profile[:,0:self.strato_lev_out,2] = y_profile[:,0:self.strato_lev_out,0].clone().zero_()
            # y_profile[:,60:60+self.strato_lev_out,0] = y_profile[:,60:60+self.strato_lev_out,0].clone().fill_(1e6)
            # y_profile[:,60:60+self.strato_lev_out,1] = y_profile[:,60:60+self.strato_lev_out,0].clone().zero_()
            # y_profile[:,60:60+self.strato_lev_out,2] = y_profile[:,60:60+self.strato_lev_out,0].clone().zero_()
        
        return y_profile
    