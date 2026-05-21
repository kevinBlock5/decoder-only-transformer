#Importing dependencies...
import torch
import torch.nn as nn

#Embedding Layer...
class EmbeddingLayer(nn.Module):
    def __init__(self,embedd_dim,vocab_size,dropout = 0.1):
        super(EmbeddingLayer,self).__init__()

        self.vocab_size = vocab_size
        self.embedd_dim = embedd_dim

        self.EL = nn.Embedding(num_embeddings = vocab_size,embedding_dim = embedd_dim,padding_idx = 0)
        self.DO = nn.Dropout(p = dropout)

    def forward(self,x):
        out = self.EL(x) * (self.embedd_dim ** 0.5)
        out = self.DO(out)

        return out

#Positional Encoding...
class PositionalEncoding(nn.Module):
    def __init__(self,embedd_dim,max_len = 1000,dropout = 0.1):
        super(PositionalEncoding,self).__init__()

        self.embedd_dim = embedd_dim
        self.max_len = max_len

        self.DO = nn.Dropout(p = dropout)
        pe = torch.zeros(max_len,embedd_dim)
        position = torch.arange(0,max_len,dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,embedd_dim,2).float() * (-torch.log(torch.tensor(10000.0)) / embedd_dim))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)
    
    def forward(self,x):
        out = x + self.pe[:,:x.size(1),:]
        out = self.DO(out)

        return out

#Multi-Head Attention...
class MultiHeadAttention(nn.Module):
    def __init__(self,embedd_dim,num_heads,dropout = 0.1):
        super(MultiHeadAttention,self).__init__()

        self.embedd_dim = embedd_dim
        self.num_heads = num_heads
        self.depth = embedd_dim // num_heads

        self.WQ = nn.Linear(in_features = embedd_dim,out_features = embedd_dim)
        self.WK = nn.Linear(in_features = embedd_dim,out_features = embedd_dim)
        self.WV = nn.Linear(in_features = embedd_dim,out_features = embedd_dim)

        self.DO = nn.Dropout(p = dropout)
        self.FL = nn.Linear(in_features = embedd_dim,out_features = embedd_dim)

    def split_heads(self,x,batch_size):
        x = x.view(batch_size,-1,self.num_heads,self.depth)
        return x.permute(0,2,1,3)
    
    def forward(self,x,mask = None):
        batch_size = x.size(0)

        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)

        Q = self.split_heads(Q,batch_size)
        K = self.split_heads(K,batch_size)
        V = self.split_heads(V,batch_size)

        scores = torch.matmul(Q,K.transpose(-2,-1)) / (self.depth ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0,float('-inf'))
        
        attn_weights = torch.softmax(scores,dim = -1)
        attn_weights = self.DO(attn_weights)

        out = torch.matmul(attn_weights,V)
        out = out.permute(0,2,1,3).contiguous().view(batch_size,-1,self.embedd_dim)
        out = self.FL(out)

        return out

#Feeforward Network...
class FeedForwardNetwork(nn.Module):
    def __init__(self,embedd_dim,ff_dim,dropout = 0.1):
        super(FeedForwardNetwork,self).__init__()

        self.embedd_dim = embedd_dim
        self.ff_dim = ff_dim

        self.FFN = nn.Sequential(
            nn.Linear(in_features = embedd_dim,out_features = ff_dim),
            nn.GELU(),
            nn.Dropout(p = dropout),
            nn.Linear(in_features = ff_dim,out_features = embedd_dim)
        )

    def forward(self,x):
        out = self.FFN(x)

        return out
    
#Decoder Block...
class DecoderBlock(nn.Module):
    def __init__(self, embedd_dim, num_heads, dropout=0.1, expansion=4):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedd_dim)
        self.mha = MultiHeadAttention(embedd_dim, num_heads, dropout)

        self.ln2 = nn.LayerNorm(embedd_dim)
        self.ffn = FeedForwardNetwork(embedd_dim, expansion * embedd_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-Norm Masked Self-Attention
        attn_out = self.mha(self.ln1(x), mask)
        x = x + self.dropout(attn_out)

        # Pre-Norm Feed Forward
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)

        return x

#Decoder Stack...
class DecoderStack(nn.Module):
    def __init__(self, num_layers, embedd_dim, num_heads, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            DecoderBlock(embedd_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)

        return x

#Head Layer...
class HeadLayer(nn.Module):
    def __init__(self,embedd_dim,vocab_size):
        super(HeadLayer,self).__init__()

        self.embedd_dim = embedd_dim
        self.output_dim = vocab_size

        self.HL = nn.Linear(in_features = embedd_dim,out_features = vocab_size)

    def forward(self,x):
        out = self.HL(x)

        return out
    
#Complete network...
class DecoderOnlyTransformer(nn.Module):
    def __init__(self,embedd_dim,vocab_size,nheads,layers,dropout,batch_first = True):
        super(Network,self).__init__()

        self.embedd_dim = embedd_dim
        self.vocab_size = vocab_size
        self.nheads = nheads
        self.layers = layers
        self.dropout = dropout
        self.batch_first = batch_first

        self.EL = EmbeddingLayer(embedd_dim = embedd_dim,vocab_size = vocab_size)
        self.PE = PositionalEncoding(embedd_dim = embedd_dim)
        self.DB = DecoderStack(num_layers = layers, embedd_dim = embedd_dim, num_heads = nheads, dropout = dropout)
        self.HL = HeadLayer(embedd_dim = embedd_dim,vocab_size = vocab_size)

    def forward(self,x,padding_mask):
        out = self.EL(x)
        out = self.PE(out)
        out = self.DB(out,padding_mask)
        out = self.HL(out)

        return out
    

if __name__ == "__main__":
    model = DecoderOnlyTransformer(
        embed_dim=512,
        vocab_size=1000,
        num_heads=8,
        num_layers=20,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {total_params:.2f}M")
