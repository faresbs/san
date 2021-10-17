"""
This was heavily inspired from http://nlp.seas.harvard.edu/2018/04/03/attention.html
"""

import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
import math, copy, time
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt

#Import customizable CNN
from models import mb2

from torchsummary import summary

#For debugging
torch.set_printoptions(threshold=5000)

###############################################################################
#
# Code for the Sign Transformer network for SLR
#
###############################################################################

#A helper function for producing N identical layers
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


#Self-Attention mechanism
def ScaledDotProductAttention(query, key, value, mask=None, dropout=None):

    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    #src_mask=(batch, 1, 1, max_seq) #NOTE: this is like the tutorials but it is weird!
    #trg_mask = (batch, 1, max_seq, max_seq)
    #score=(batch, n_heads, Seq, Seq)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    output = torch.matmul(p_attn, value)

    return output #(Batch, n_heads, Seq, d_k)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.3):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units
        """
        super(MultiHeadedAttention, self).__init__()

        # This sets the size of the keys, values, and queries (self.d_k) to all 
        # be equal to the number of output units divided by the number of heads.
        #d_k = dim of key for one head
        self.d_k = n_units // n_heads

        #This requires the number of n_heads to evenly divide n_units.
        #NOTE: nb of n_units (hidden_size) must be a multiple of 6 (n_heads) 
        assert n_units % n_heads == 0
        #n_units represent total of units for all the heads

        self.n_units = n_units
        self.n_heads = n_heads

        self.linears = clones(nn.Linear(n_units, n_units), 4)

        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
                    [l(x).view(nbatches, -1, self.n_heads, self.d_k).transpose(1, 2)
                    for l, x in zip(self.linears, (query, key, value))]


        # 2) Apply attention on all the projected vectors in batch.
        x = ScaledDotProductAttention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.n_heads * self.d_k)

        z = self.linears[-1](x)

        #(batch_size, seq_len, self.n_units)
        return z

#-------------------------------------------------------------------------------

#Extract embeddings from hand images
#Hand representations

#TO DO: to customize like 2D_embeddings
class HandExtractor(nn.Module):
    def __init__(self, n_units=1280, pretrained=False, network_type='mb2', channels=1):
        super(HandExtractor, self).__init__()

        #self.network = torchvision.models.mobilenet_v2(pretrained=pretrained)
        self.network = mb2.mobilenet_v2(pretrained=pretrained, channels=channels)

        #Drop FC layer
        modules = list(self.network.children())[:-1]
        self.network = nn.Sequential(*modules)

        self.n_units= n_units

    #Input (batch, hands, seq, 3, 64, 64)
    #Input 1 hand (batch, seq, 3, 64, 64)
    def forward(self, x):
        batch, seq, _, _, _ = x.shape

        x = x.view(batch*seq, x.shape[-3], x.shape[-2], x.shape[-1])
        emb = self.network(x)

        #Apply AVG POOL if we have a feature map
        if(len(emb.shape) > 2):
            emb =  emb.mean(3).mean(2)

        #Reshape embeddings as expected from transformer block
        emb = emb.view(batch, -1, self.n_units)

        return emb

#----------------------------------------------------------------------------------

#Extract embeddings from images
#Full frame representations

class src_2Dembeddings(nn.Module):
    def __init__(self, n_units, pretrained=True, image_size=224, network_type='mb2', channels=3):

        super(src_2Dembeddings, self).__init__()

        self.n_units = n_units
        self.network_type = network_type

        self.position = PositionalEncoding(n_units, 0.3)

        #Use mb2 for image embeddings
        #TO DO: customize other arch to make them able train on grayscale
        if(self.network_type=='mb2'):
            #self.network = torchvision.models.mobilenet_v2(pretrained=pretrained)
            self.network = mb2.mobilenet_v2(channels=channels, pretrained=pretrained)

        elif (self.network_type=='alexnet'):
            self.network = torchvision.models.alexnet(pretrained=pretrained)

        elif (self.network_type=='resnet'):
            self.network = torchvision.models.resnet18(pretrained=pretrained)

        elif (self.network_type=='resnext'):
            self.network = torchvision.models.resnext50_32x4d(pretrained=pretrained)

        elif(self.network_type=='wide_resnet'):
            self.network = torchvision.models.wide_resnet50_2(pretrained=pretrained)

        else:
            print('No supported architecture!!')
            quit(0)

        #Drop FC layer
        modules = list(self.network.children())[:-1]
        self.network = nn.Sequential(*modules)

        #Placeholder for gradients
        self.gradients = None

    #hook for gradientrs of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        #Reshape to join seq size w/ batch
        x = x.view(batch_size*seq_len, x.shape[2], x.shape[3], x.shape[4])

        #Extract embeddings for full frames
        feature_map = self.network(x)

        #register the hook
        #if(feature_map.requires_grad):
            #print('here')
         #   feature_map.register_hook(self.activations_hook)
            #sd

        #Apply AVG POOL if we have a feature map
        if(len(feature_map.shape) > 2):
            frame_embeddings =  feature_map.mean(3).mean(2)

        #Reshape embeddings as expected from transformer block
        frame_embeddings = frame_embeddings.view(batch_size, -1, self.n_units)

        #image emb = (batch, seq_length, n_units)
        return frame_embeddings, feature_map, self.gradients


#The positional encoding 'tags' each element of an input sequence with a code that 
#identifies it's position (i.e. time-step).
#We use sine and cosine functions to encode positions

class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0., max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0., n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):

        #Positianal encoding (batch, seq_length, emb_size)
        pos = Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x + pos)

##########

#Encoder stack is composed of a multi-head self attention + norm layer + FF + norm
#See figure's left side from original transformner network
class EncoderStack(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderStack, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        #2 skip+norm layers: one after muli-head attention, the second after FF
        self.sublayer = clones(ResidualSkipWithLayerNorm(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None, hand_emb=None):

        #Pass hand embeddings as query, full frame embeddings as context
        if(type(hand_emb) != type(None)):
            x = self.sublayer[0](hand_emb, None, lambda hand_emb: self.self_attn(hand_emb, x, x, mask))

        #If there is no query: query, keys and vector are the same
        else:
            x = self.sublayer[0](x, None, lambda x: self.self_attn(x, x, x, mask))

        return self.sublayer[1](x, None, self.feed_forward)


#Encoder network is composed of N encoder stacks (layers)

##You want them to have different parameters; the point of stacking
#multiple encoders is that each encoder can transform the data independently
#of the others, resulting in a more expressive model.
#If they have the same parameters, then it's the same as having 1 encoder.

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, hand_emb, mask):

        #Pass the input (and mask) through each layer in turn.
        for i in range(0, len(self.layers)):
            x = self.layers[i](x, mask, hand_emb)

            if(type(hand_emb) != type(None)):
                break

        #(batch, seq, n_units)
        return self.norm(x)

##########

#Produce the output probablitites
#linear layer + softmax
class Output_layer(nn.Module):

    def __init__(self, d_model, vocab_size):
        super(Output_layer, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

##########

#Full transformer network architecture (encoder + decoder + output)
class FullTransformer(nn.Module):
    def __init__(self, encoder, src_emb, output, position):
        super(FullTransformer, self).__init__()
        self.encoder = encoder
        self.src_emb = src_emb
        self.output_layer = output
        self.hand_emb = HandExtractor()
        self.position = position

        #Placeholder for gradients
        self.gradients = None

        #Placeholder for activations
        self.activations = None

     #method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    #method for the activation extraction
    def get_activations(self):
        return self.activations

    #hook for gradientrs of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def encode(self, src_emb, hand_emb, src_mask):
        return self.encoder(src_emb, hand_emb, src_mask)

    #Call this when training
    def forward(self, src, src_mask, rel_mask=None, hand_seqs=None, arch='CNN-attention-CTC'):

        #Use normal mask
        if(type(rel_mask) == type(None)):
            rel_mask = src_mask

        #Get context seq emb
        src_emb, f_map, grad = self.src_emb(src)

        #register the hook
        if(f_map.requires_grad):
            f_map.register_hook(self.activations_hook)

        self.activations = f_map

        if(arch=='CNN-attention-CTC'):
            #(batch, seq_length, feature_dim)
            src_emb = self.position(src_emb)
            src_emb = self.encode(src_emb, None, src_mask)

        full_out = self.output_layer(src_emb)

        #Get hand seq emb
        if(type(hand_seqs) != type(None)):

            hand_emb = self.hand_emb(hand_seqs)

            if(arch=='CNN-attention-CTC'):
                hand_emb = self.position(hand_emb)
                hand_emb = self.encode(hand_emb, None, src_mask)

            #Context-Hand attention
            #Combine hand emb with its context
            #Use relative masking
            comb_emb = self.encode(src_emb, hand_emb, rel_mask)

            hand_out = self.output_layer(hand_emb)
            comb_out = self.output_layer(comb_emb)

        else:
            comb_out = None
            hand_out = None

        return comb_out, full_out, hand_out


#Create the full model
def make_model(tgt_vocab, n_stacks=3, n_units=512, n_heads=2, d_ff=2048, dropout=0.3, image_size=224, pretrained=True, emb_type='2d', emb_network='mb2', full_pretrained=None, hand_pretrained=None, freeze_cnn=False, channels=3):

    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units, dropout)
    ff = PositionWise(n_units, d_ff, dropout)
    position = PositionalEncoding(n_units, dropout)

    model = FullTransformer(
        Encoder(EncoderStack(n_units, c(attn), c(ff), dropout), n_stacks),
        src_2Dembeddings(n_units, pretrained, image_size, network_type=emb_network, channels=channels),
        Output_layer(n_units, tgt_vocab),
        c(position)
        )

    #Load pretrained CNNs (trained on same dataset)
    if(full_pretrained):
        model.src_emb.load_state_dict(torch.load(full_pretrained))
        print("Full frame CNN pretrained weights successfully loaded..")

    if(hand_pretrained):
        model.hand_emb.load_state_dict(torch.load(hand_pretrained))
        print("Hand CNN pretrained weights succesfully loaded..")

    #Initialize parameters with Glorot/xavier. (except image/hand embeddings that are init by imagenet weights)
    for name, p in model.named_parameters():

        if(freeze_cnn and ('src_emb' in name or 'hand_emb' in name)):
            p.requires_grad = False

        #Fan in/out can't compute with dim > 1
        if p.dim() > 1 and 'src_emb' not in name and 'hand_emb' not in name:
            nn.init.xavier_uniform_(p)

    return model


#########################


#----------------------------------------------------------------------------------
#Layer normalization
class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, query, sublayer):
        "Apply residual connection to any sublayer with the same size."
        attn_x = self.dropout(sublayer(self.norm(x)))
        return x + attn_x

#A simple Feed forward MLP
class PositionWise(nn.Module):

    def __init__(self, n_units, d_ff, dropout=0.1):
        super(PositionWise, self).__init__()
        self.w_1 = nn.Linear(n_units, d_ff)
        self.w_2 = nn.Linear(d_ff, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


