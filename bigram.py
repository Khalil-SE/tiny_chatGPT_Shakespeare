import torch
import torch.nn as nn
import torch.nn.functional as F


# Hyperparameters
batch_size = 64 # How many sequences will we process in parallel?
block_size = 256 # What is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout_rate = 0.2

torch.manual_seed(1337) # Seed for reproducibility

with open ('tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create a set of all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Creating the mapping from chars to ints and back
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
# Split train and test
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
        wei = F.softmax(wei, dim=-1) # (B,T,T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_rate),
        )


    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(num_heads=n_head, head_size=head_size) # Communication: Multi-Head Self Attention
        self.ffwd = FeedForward(n_embd) # Computation: Feed Forward Layer
        self.ln1 = nn.LayerNorm(n_embd) # Layer Normalization
        self.ln2 = nn.LayerNorm(n_embd) # Layer Normalization

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x)) # apply multi-head attention and add residual connection with layer normalization
        x = x + self.ffwd(self.ln2(x))     # apply feed forward layer and add residual connection with layer normalization
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.sa_head = Head(n_embd) # Self Attention Head
        # self.sa_heads = MultiHeadAttention(num_heads=4, head_size=n_embd//4) # Multi Head Attention i.e 4 heads of 8-dimensional self-attention
        # self.ffwd = FeedForward(n_embd) # Feed Forward Layer
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     # Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd)
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.nlf = nn.LayerNorm(n_embd) # Final Layer Normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target=None):
        B, T = idx.shape
        
        # idx and target both are (B,T) tensor of integer
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_embd + pos_emb # (B,T,C)
        # x = self.sa_heads(x) # apply multiple heads of self-attention (B,T,C)
        # x = self.ffwd(x) # (B, T, C) apply the feed forward layer
        x = self.blocks(x) # (B, T, C) apply the transformer blocks
        x = self.nlf(x) # (B, T, C) apply final layer normalization
        logits = self.lm_head(x) # (B,T,vocab_size)
        if target == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] # (B, block_size)
            # get the predictions
            logits, loss = self(idx_cond)
            # focus on the only last time step
            logits = logits[:, -1, :] #become (B, C)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) #(B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            # append sampled index to the runing sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# Create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(idx = context, max_new_tokens=300)[0].tolist()))