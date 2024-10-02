import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

with open("input.txt", "r", encoding="utf-8") as f:
    textdata = f.read()

vocab = sorted(list(set(textdata)))
vocab_size = len(vocab)


ctoi = {c: i for i, c in enumerate(vocab)}
itoc = {i: c for c, i in ctoi.items()}
encode = lambda s: [ctoi[i] for i in s]
decode = lambda l: "".join([itoc[i] for i in l])

dataset = torch.tensor(encode(textdata))

n = int(0.9 * len(dataset))

x_train = dataset[:n]
x_val = dataset[n:]


# hyperparameters

batch_size = 64
context_size = 256
learning_rate = 3e-3
epochs = 2000
n_embed = 512
eval_iters = 200
eval_interval = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
num_heads = 8
num_layers = 6
head_size = n_embed // num_heads
dropout = 0.2


# data loading
def get_data(split):
    data = x_train if split == "train" else x_val
    ix = torch.randint(len(data) - batch_size, (batch_size,))
    x = torch.stack([data[i : i + context_size] for i in ix])
    y = torch.stack([data[i + 1 : i + context_size + 1] for i in ix])
    return x.to(device=device), y.to(device=device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_data(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)  # n_embed * head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)  # n_embed * head_size
        self.value = nn.Linear(n_embed, head_size, bias=False)  # n_embed * head_size
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # x => B * T * n_embed
        q = self.query(x)  # B * T * head_size
        k = self.key(x)  # B * T * head_size
        v = self.value(x)  # B * T * head_size
        wei = q @ k.transpose(-2, -1) * C**-0.5  # B * T * T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # B * T * T
        wei = self.dropout(wei)
        out = wei @ v  # B * T * head_size
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.f_layer = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),  # non linearity
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.f_layer(x)  # B * T * n_embed


class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for x in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # B * T * (head_size * num_heads)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    def __init__(self, num_heads, n_embed):
        super().__init__()
        self.sa = MultiHead(num_heads=num_heads, head_size=head_size)  # B * T * n_embed
        self.ffwd = FeedForward(n_embed=n_embed)  # B * T * n_embed

    def forward(self, x):
        x = x + self.sa(x)  # B * T * n_embed
        x = x + self.ffwd(x)  # B * T * n_embed
        return x


class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_size, n_embed)
        # self.sa_head = Hes
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.blocks = nn.Sequential(
            *[Block(num_heads=num_heads, n_embed=n_embed) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # B * T * n_embed
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # T * n_embed
        x = tok_emb + pos_emb  # B * T * n_embed
        x = self.blocks(x)
        x = self.dropout(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy(logits.permute(0, 2, 1), target=targets)

        return logits, loss

    def generate(self, idx, max_num_tokens):
        for i in range(max_num_tokens):

            idx_cond = idx[:, -context_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_idx), dim=-1)
        return idx


model = BigramModel()

print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

optimzer = torch.optim.AdamW(lr=learning_rate, params=model.parameters())

loss_history = []
epochs_i = []

for epoch in range(epochs):

    # every once in a while evaluate the loss on train and val sets
    if epoch % eval_interval == 0 or epoch == epochs - 1:
        losses = estimate_loss()
        print(
            f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_data("train")
    logits, loss = model(xb, yb)

    optimzer.zero_grad(set_to_none=True)

    loss.backward()

    optimzer.step()
    # if epoch // 100 == 0:
    #     print(f"loss => {loss.item()}")
    #     loss_history.append(loss.item())
    #     epochs_i.append(epoch)


print(
    decode(
        model.generate(torch.zeros((1, 1), dtype=torch.long), max_num_tokens=300)[
            0
        ].tolist()
    )
)
