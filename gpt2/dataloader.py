import torch
import tiktoken


class DataLoader:
    def __init__(self, B, T, device):
        self.B = B
        self.T = T

        with open("/kaggle/working/input.txt", "r", encoding="utf-8") as f:
            data = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(data)

        self.tokens = torch.tensor(tokens)

        self.current_start_position = 0
        self.device = device

    def next_batch(self):
        B, T = self.B, self.T
        buff = self.tokens[
            self.current_start_position : self.current_start_position + B * T + 1
        ]
        buff = buff.to(self.device)

        # train data inputs
        x = (buff[:-1]).view(B, T)
        # target lebels
        y = (buff[1:]).view(B, T)

        self.current_start_position += B * T

        if self.current_start_position + (B * T + 1) > len(self.tokens):
            self.current_start_position = 0

        return x, y
