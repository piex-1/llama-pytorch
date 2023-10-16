import torch
import torch.nn  as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

# some necessary parameters
# @dataclass represent some __inil__ way to simplify the all wayto define class
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers : int = 32
    n_heads: int = 32# number of heads for Query
    n_kv_heads: Optional[int] = None #Number of heads for the K and V 
    vocab_size : int = -1 # siganl ,will be set when we load the taken

    # the following  2 represents the feed-forwards layers隐藏层参数
    multiple_of :int = 256
    ffn_dim_multiplier: Optional[float] = None

    #norm learning rate
    norm_eps: float = 1e-5

    # for KV layers
    max_batch_size :int =32
    max_seq_len :int = 2048

    # examples:  CPU CUDA 
    device: str = None

# 10000.0  from papers
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int ,device: str, theta: float = 10000.0):
    assert head_dim % 2  == 0, "this way can not applied on  evn number, as the paper say"

    # bulid theta parameter
    # according to the formula ，且使用是在分割成多heads后才继续进行，然后shape = head_dim /2 
    # theta_i = 10000 * (-2(i-1)/dim) for 1 2 3...dim/2 ,
    theta_numerator = torch.arange(0, head_dim, 2).float() # 构造了一个tensor，【0，2，4，...直到小于dim

    # Shape : head_demension/2
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)# CPu or GPU 

    # Shape : Seq_len  构造张量‘m'
    m = torch.arange(seq_len, device = device)
    
    # m theta 的外积
    freqs = torch.outer(m, theta).float()
    # c = R * exp(m * theta), 这里 R = 1 :
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    # polar 转化复数，
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

# 处理旋转bedding，将 freqs_complex处理
def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # 复数分离实部和虚部
    # (Batch_dim, Seq_Len, Head_dim, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. 两个额外batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)# 一定要让他保持一样的形状
    return x_out.type_as(x).to(device)

# RMS nomolization 
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps# 分母中，防止除以0       
        self.weight = nn.Parameter(torch.ones(dim))# The gamma parameter

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        # mean -1 最后dimension均值，保持结果维度不变
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)# 分母中，防止除以0  

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)# args.dim输入的特征数量，the latter one is output dim
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(   
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_complex: torch.Tensor
    ):
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xv = self.wv(x)

        # 按照要求分裂heads  (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        #  ralative position encoding，提前计算好就行可以了
        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # from cache 得到 K，V
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.

        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)扩充到self.n_rep
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # 重新链接所有块为整体(B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, Dim) -> (B, 1, Dim)

class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelArgs
    ):
        super().__init__()

        # 确定隐藏层数
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:# 如果指定的话，  still has a multiplier
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter     "multiple_of": 256
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x

# 就是重复N 次的block
class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads  # 查看你parameter 的参数得到结果4096
        self.dim = args.dim # 查看你parameter 32
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE the attention block
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()# 夫类的构造函数，以确保正确初始化

        # 确保vocab_size
        assert args.vocab_size != -1, "vocabulary size must be set"
         
        self.args = args # save
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers # 重复的layer,N 次重复
        self.tok_embeddings =  nn.Embedding(self.vocab_size, args.dim)

        # N 次重复, encodeer
        self.layers  =  nn.ModuleList()
        for _ in range (args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        # 对照Rmsnorm过程
        self.norm = RMSNorm(args.dim, eps= args.norm_eps)# with feather size
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # 提前计算旋转编码的频率
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

        # taken =1 due to the KV cache,
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # will be like BATCH + Seq_Len
        batch_size , seq_len = tokens.shape
        assert seq_len == 1,"Only one taken/time can be processed"

        # (B,seq_len) => (B, Seq_len, Dim)
        # dim = dimension = 4096
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        # 提前计算position 
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # 连续apply on all layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)# 连续运行多次大block的操作
        h = self.norm(h)
        output = self.output(h).float()

        return output

