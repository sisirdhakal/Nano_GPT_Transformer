# My GPT-2 Implementation

## Papers 📄

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/papers/language_models_are_unsupervised_multitask_learners.pdf)
- [Language Models are Few-Shot Learners](https://cdn.openai.com/papers/language_models_are_few_shot_learners.pdf)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/2001.08361)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2006.05837)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2205.14135)
- [Online Normalizer Calculation for Softmax](https://arxiv.org/abs/2301.02562)

## Acknowledgments 🙏

A special thanks to Andrej Karpathy for the inspiration and resources that guided my understanding of the GPT architecture.

## Key Learnings 💡

- **Weight Initialization**: Understanding the significance of initializing weights based on vocabulary size, where the expected loss for random initialization is approximately 10.82.
- **Gradient Accumulation**: Implemented techniques to optimize training by accumulating gradients over multiple batches, which improves memory efficiency.
- **Torch Compile**: Learned how to leverage `torch.compile` for optimizing the model's performance during training.
- **Mixed Precision Training**: Gained insights into using bfloat16 (BF16) for faster training without significant loss in model accuracy.
- **Flash Attention**: Explored the implementation of Flash Attention to enhance computational efficiency and memory usage during training.
- **Normalization Clipping**: Implemented normalization clipping strategies to stabilize training and mitigate gradient shocks during batch updates.

## Important Notes 🍀

- The input and output embeddings are tied together, sharing the same 2D tensor shape and data pointer, which helps in improving the model's efficiency and performance.
- Observing the smoothness of positional and token embedding weights can indicate the convergence status of the model. Fluctuations may suggest insufficient training.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
