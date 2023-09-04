import paddle
import paddle.nn as nn
from paddle.nn import Linear, Dropout, LayerNorm, LayerList, Layer
from paddle.static import InputSpec
from paddle.nn import functional as F

# @paddle.incubate.passes.ir.RegisterPass
# def generate_delete_dropout():
#     def pattern(x):
#         dropout_op = paddle.incubate.passes.ir.PassDesc.OP.dropout
#         dropout_op._inputs.pop("Seed")
#         dropout_op._outputs.pop("Mask")
#         return dropout_op(X=x)

#     def replace(x):
#         return paddle.scale(x)

#     return pattern, replace

# # @paddle.incubate.passes.ir.RegisterPass(input_specs={'x': InputSpec([-1, 384, 768]), 'attn_mask': InputSpec([-1, 12, 384, 384])})
# @paddle.incubate.passes.ir.RegisterPass(input_specs={"q": InputSpec([8, 16, 384, 64]),
#                                                      "k": InputSpec([8, 16, 384, 64]),
#                                                      "v": InputSpec([8, 16, 384, 64]),
#                                                      "attn_mask": InputSpec([8, 16, 384, 384])})
# def generate_fused_multihead_attention():
#     def pattern(q, k, v, attn_mask):
#         add_residual = True
#         pre_ln = True
#         attn_dropout = 0.1
#         # add_mask = True
#         add_mask = False
#         batch_size = 8
#         seq_len = 384
#         hidden_size = 1024
#         num_heads = 16
#         sdp = SDP(
#             hidden_size,
#             num_heads,
#             # add_residual=add_residual,
#             # pre_ln=pre_ln,
#             attn_dropout=attn_dropout,
#         )
#         return sdp(q, k, v, attn_mask)

#     def replace(q, k, v, attn_mask):
#         fuse_op = paddle.incubate.passes.ir.PassDesc.OP.multihead_attention(
#             #X=x,
#             Q=q,
#             K=k,
#             V=v,
#             Attn_mask=attn_mask
#         )
#         return fuse_op

#     return pattern, replace

# class SDP(Layer):
#     def __init__(self,
#                  embed_dim,
#                  num_heads,
#                  attn_dropout=-1):
#         super(SDP, self).__init__()    
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
#         self.attn_dropout = attn_dropout
        
#     def forward(self, q, k, v, attn_mask=None):
#         # compute scale dot prod
#         product = paddle.matmul(x=q, y=k, transpose_y=True)
#         print('matmul q=', q.size(), ', k=', k.size())
#         product = paddle.scale(product, scale=self.head_dim**-1.5)
#         if attn_mask is not None:
#             print('qk mask')
#             product = product + attn_mask

#         weights = F.softmax(product)
#         print('qk softmax')
#         # out = weights
#         # if enable dropout can't work
#         if self.attn_dropout:
#             print('qk dropout')
#             weights = paddle.scale(weights, -1.9)

#         out = paddle.matmul(weights, v)
#         print('matmul (qk)=', weights.size(), ', v=', v.size()))
#         return out       
        
        
