import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch import nn
import inspect


""" LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """


class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # ensuring the number of embeddings (n_embed) is divisible by the number of heads (n_head)
        assert config.n_embed % config.n_head == 0

        # defining a linear layer for computing queries, keys, and values from input embeddings
        self.c_attn = nn.Linear(
            config.n_embed, 3 * config.n_embed, bias=config.bias
        )  # output shape: b * t * (3 * n_embed)
        # defining a linear layer for projecting the output back to the original embedding size
        self.c_proj = nn.Linear(
            config.n_embed, config.n_embed, bias=config.bias
        )  # output shape: b * t * n_embed
        self.c_proj.GPT2_SCALE = 1.0

        # storing the number of heads and embedding size for future use
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # creating a lower triangular matrix for masking to prevent attending to future tokens
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.context_size, config.context_size)).view(
                1, 1, config.context_size, config.context_size
            ),  # shape: (1, 1, context_size, context_size)
        )

    def forward(self, x):
        b, t, c = (
            x.shape
        )  # extracting batch size (b), sequence length (t), and embedding size (c)

        # computing queries, keys, and values using the linear layer
        qkv = self.c_attn(x)  # shape: b * t * (3 * n_embed)
        # splitting the output into queries (q), keys (k), and values (v)
        q, k, v = qkv.split(self.n_embed, dim=2)  # each has shape: b * t * n_embed
        # reshaping and transposing to prepare for multi-head attention
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(
            1, 2
        )  # shape: b * n_head * t * (n_embed / n_head)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(
            1, 2
        )  # shape: b * n_head * t * (n_embed / n_head)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(
            1, 2
        )  # shape: b * n_head * t * (n_embed / n_head)

        #         # calculating attention scores using scaled dot-product
        #         wei = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))  # shape: b * n_head * t * t
        #         # applying masking to prevent attention to future tokens
        #         wei = wei.masked_fill(self.bias[:, :, :t, :t] == 0, float("-inf"))  # shape: b * n_head * t * t
        #         # normalizing attention weights using softmax
        #         wei = F.softmax(wei, dim=-1)  # shape: b * n_head * t * t
        #         # multiplying attention weights by values to get the output
        #         out = wei @ v  # shape: b * n_head * t * (n_embed / n_head)

        #       flash attention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # reshaping back to original dimensions
        out = out.transpose(1, 2).contiguous().view(b, t, c)  # shape: b * t * n_embed
        # projecting the output back to embedding size
        out = self.c_proj(out)  # shape: b * t * n_embed
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # defining a linear layer for expanding the input size
        self.c_fc = nn.Linear(
            config.n_embed, 4 * config.n_embed, bias=config.bias
        )  # shape: b * t * (4 * n_embed)
        # defining the gelu activation function for non-linearity
        self.gelu = nn.GELU(approximate="tanh")  # gelu activation function
        # defining a linear layer for reducing back to the original embedding size
        self.c_proj = nn.Linear(
            4 * config.n_embed, config.n_embed, bias=config.bias
        )  # shape: b * t * n_embed
        self.c_proj.GPT2_SCALE = 1.0

    def forward(self, x):
        # expanding the dimensionality using the first linear layer
        x = self.c_fc(x)  # shape: b * t * (4 * n_embed)
        # applying the activation function
        x = self.gelu(x)  # shape: b * t * (4 * n_embed)
        # reducing back to the original embedding size
        x = self.c_proj(x)  # shape: b * t * n_embed
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # defining layer normalization for inputs to the attention layer
        self.ln_1 = LayerNorm(
            config.n_embed, bias=config.bias
        )  # normalizing input to the attention layer
        # instantiating the causal self-attention layer
        self.attn = CausalSelfAttention(config)  # causal self-attention layer
        # defining layer normalization for inputs to the mlp
        self.ln_2 = LayerNorm(
            config.n_embed, bias=config.bias
        )  # normalizing input to the mlp
        # instantiating the feedforward mlp
        self.mlp = MLP(config)  # feedforward mlp

    def forward(self, x):
        # adding the output of the attention layer to the input (skip connection)
        x = x + self.attn(self.ln_1(x))  # shape: b * t * n_embed
        # adding the output of the mlp to the input (skip connection)
        x = x + self.mlp(self.ln_2(x))  # shape: b * t * n_embed
        return x  # returning the final output of the block


@dataclass
class ChatGptConfig:
    n_layer: int = 12  # setting the number of transformer layers
    n_head: int = 12  # setting the number of attention heads
    n_embed: int = 768  # setting the size of the embeddings
    context_size: int = 1024  # setting the maximum length of input sequences
    vocab_size: int = 50304  # setting the number of unique tokens in the vocabulary
    bias: bool = True


class Gpt2(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_size is not None
        self.config = config
        # defining the main components of the gpt-2 model
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.vocab_size, config.n_embed
                ),  # defining token embeddings (b * t * n_embed)
                wpe=nn.Embedding(
                    config.context_size, config.n_embed
                ),  # defining positional embeddings (b * t * n_embed)
                h=nn.ModuleList(
                    Block(config) for _ in range(config.n_layer)
                ),  # stacking transformer blocks
                ln_f=LayerNorm(
                    config.n_embed, bias=config.bias
                ),  # final layer normalization (b * t * n_embed)
            )
        )
        # defining a linear layer for generating output logits from final embeddings
        self.lm_head = nn.Linear(
            config.n_embed, config.vocab_size, bias=False
        )  # shape: b * t * vocab_size

        # sharing of the weights
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    # initialization of the weights based on the gpt 2
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "GPT2_SCALE"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        assert (
            t <= self.config.context_size
        ), f" tokens {t} cannot be more than context window size {self.config.context_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)  # shape: (t)
        post_emb = self.transformer.wpe(pos)  # shape : (t * n_embed)
        tkn_embd = self.transformer.wte(idx)  # shape : (b * t * n_embed)
        x = tkn_embd + post_emb  # shape : (b * t * n_embed)

        # forwarding the x in each block
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)  # shape : (b * t * n_embed)
        logits = self.lm_head(x)  # shape : (b * t * vocab_size)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @classmethod
    def from_pretrained(cls, model_type):
        # ensuring the model type is one of the predefined options
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        # defining configuration mapping for different model types
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600),
        }[model_type]

        print(f"Loading pretrained weights for the model {model_type}")

        config_args["vocab_size"] = 50257  # setting vocabulary size for the model
        config_args["context_size"] = (
            1024  # setting the maximum context size for the model
        )

        # creating a configuration object for the gpt-2 model
        config = ChatGptConfig(**config_args)

        model = Gpt2(
            config
        )  # instantiating the gpt-2 model with the specified configuration
        sd = model.state_dict()  # getting the state dictionary of the model
        sd_keys = sd.keys()
        # excluding bias keys that are not part of the model parameters
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # loading the pre-trained model from hugging face transformers library
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # excluding unnecessary keys from the hugging face model's state dictionary
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignoring masked bias
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # ignoring attention bias

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # hardcoding to transpose weights for certain layers since gpt uses a "Conv1D" module instead of a standard Linear layer
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"ensuring keys match: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # transposing weights for specific layers to match expected shapes
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # copying over parameters for other layers
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
