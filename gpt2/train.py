# model = Gpt2.from_pretrained('gpt2')
import time
from dataloader import DataLoader
import torch
from model import Gpt2, ChatGptConfig

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


total_batch_size = 524288  # number with more 2's power
B = 8
T = 512

assert total_batch_size % (B * T) == 0

grad_accum_steps = total_batch_size // (B * T)


train_loader = DataLoader(B=B, T=T, device=device)

# converting to bfloat16 for higher performance
torch.set_float32_matmul_precision("high")

# schedulers for the learning rate
# max_lr = 6e-4
max_lr = 3e-4
min_lr = max_lr * 0.1
initial_eochs = 5
max_epochs = 2


def get_lr(it):
    return max_lr


#     # 1) linear warmup for initial_epochs steps
#     if it < initial_eochs:
#         return max_lr * it / initial_eochs
#     # 2) if it > lr_decay_iters, return min learning rate
#     if it > max_epochs:
#         return min_lr
#     # 3) in between, use cosine decay down to min learning rate
#     decay_ratio = (it - initial_eochs) / (max_epochs - initial_eochs)
#     assert 0 <= decay_ratio <= 1
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
#     return min_lr + coeff * (max_lr - min_lr)


model = Gpt2(
    ChatGptConfig(vocab_size=50304)
)  # tokens padded up to nearest number with more 2's power
model = model.to(device)
# torch.compile is really great for running the model fast
# model = torch.compile(model)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8 )
optimizer = model.configure_optimizers(
    weight_decay=0.1, learning_rate=max_lr, betas=(0.9, 0.95), device_type=device
)

for epoch in range(max_epochs):
    t0 = time.time()

    optimizer.zero_grad()

    # gradient accumulation to match the batch size

    losses_accum = 0

    for grad_accum in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # autocasting to bfloat16 for faster performance
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
            # scaling the loss to account for the gradient accumulation
            # gradients are just summed at each loss.backward() but since we want MEAN we devide by the loss/ number of steps
            # as in cross entropy we want the mean i.e. reduction
            loss = loss / grad_accum_steps
            losses_accum += loss.detach()
        loss.backward()
    # normalization so that the model doesnot face shock due to data misbalance in the batch
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #   learning rate setting
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    # just for to wait till the gpu's and cpu's are synchronized
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)

    print(
        f"epoch: {epoch} | loss: {losses_accum.item():.5f} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec}"
    )


# # calculating the total number of parameters
# num_parameters = sum(p.numel() for p in model.parameters())
# print(f'number of parameters: {num_parameters // 1_000_000}M')
