# GPT 2.0 from Scratch

This repository contains the implementation of a 124M parameter GPT architecture trained on 'The Verdict' by Edith Wharton.


* Author: Vidhant Maini
* Timeline: Late 2025 - Mid 2026

## Features
- **Hardware-Agnostic:** Support for CUDA, MPS (Apple Silicon), and CPU.
- **Efficient Data Loading:** Implemented a sliding window buffer with configurable stride (currently 256 for zero-overlap).
- **Advanced Metrics:** Integrated Perplexity tracking ($e^{Loss}$) for better interpretability of model confidence.

## Input Data

* To make it possible to train the model on my personal Macbook Pro, I used a toy dataset - 'The Verdict' by Edith Wharton containing 5120 tokens. However, the same code should work for larger models as well (only that it'll take more time!)
* The train-validation split was 90-10.

## Model Architecture

![](images/gpt-2-architecture.png)

https://excalidraw.com/#json=kEtqwgDj8AgcazuEOskfg,y6R5g3fUyITXRsBLGTsqIQ

## Key Params

* 124M parameters
* 12 transformer layers
* 12 multi head attention modules within each transformer
* GeLU activation for the Feed Forward Network
* Context window: 256 tokens
* Optimization: AdamW with a learning rate of 0.0004 and 0.1 weight decay.
* Embedding dimension: 768
* Stride: 256 (Zero-overlap for maximum data efficiency).
* Tokenizer: Tiktoken (`tiktoken.get_encoding("gpt2")`)

## Training Performance
The model was trained for 10 epochs using the AdamW optimizer.

It took 13 minutes on my 2019, 1.4 GHz Quad-Core Intel Core i5 Macbook Pro

![](images/model-training-6.png)

![](images/model-training-7.png)


## Sample Output
* **Prompt:** "Every effort moves you"
* **Generated:** "Every effort moves you?" "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"

Yes, the generated output doesn't quite match up to a modern LLM, but:
* the architecure follows the real GPT-2.0 closely
* was trained on a much smaller dataset to make it possible on my personal macbook pro
* is not instruction tuned

## Experiments

### 1. KV Caching

#### KV Caching makes inference faster

Compared the inference time, with and without KV-caching, for a small 124M parameter GPT-2 model.

![](images/experiments-1.png)
Observations:
* For the standard method (non-KV cache): The time taken to generate each subsequent token increases linearly because the model must re-process the entire growing sequence (prompt + previously generated tokens) at every step. This leads to $O(n^2)$ complexity relative to the sequence length.
* For the KV Cache method: The time to generate each token remains nearly constant. By storing and reusing the Key ($K$) and Value ($V$) tensors for past tokens, the model only needs to compute the $Q, K, V$ for the single new token, maintaining $O(n)$ complexity.
* Theoretically, the performance gains from KV caching become more significant as the generation length increases
* However, there is an anomaly based on our empirical evaluation - the performance gain from prompt=200, gen=50 tokens (8.94x) is more than the gain from prompt=50, gen=200 tokens (7.08x). Ideally, we would expect the opposite - as we generate more and more tokens, KV caching should shine more and more, however, in this case that is not the case. This is likely because of the per-step overhead in the decode step for things like: Python loop iteration, memory allocation for the growing KV cache, etc. KV-cache's FLOP cost per step is small, so the fixed overhead is a larger fraction of each step's total cost. This dilutes the speedup ratio when the number of generated tokens ($G$) is large.

Downsides
* The standard method is compute-bound (doing the same math over and over), whereas KV caching becomes memory-bandwidth bound (moving the stored cache from VRAM to the processor). Solution - Flash Attention, which I try next.

| Feature | Standard Inference | KV-Cache Inference |
|---|---|---|
| Computation | Recomputes all previous tokens | Computes only the new token |
| Time per Token | Increases per step | Constant |
| Complexity | $O(n^2)$ | $O(n)$ |
| Primary Bottleneck | GPU Compute (FLOPs) | Memory Bandwidth (IO) |
---

#### KV Cache costs Memory

##### Theoretical size for the 124M param model
Per token = `2 tensors (K+V) * 12 layers * 12 heads * 768/12 embedding dimension per head * 4 bytes per float = 72 KB / token`

For full context (256 tokens) = `72KB * 256 = 18 MB (approx)`

![](images/experiments-2.png)

* The KV cache is fully predictable
* KV cache size grows linearly with sequence length, up to a maximum of `context_length` tokens.

## Attention Module

1. Implemented **simple self-attention** on a sample input
    * Computed attention scores (Q@K.transpose)
    * Computed attention weights using (softmax(attn_score/k_d**-0.5))
    * Computed context vectors using attn_weights @ V
2. Next, I implemented simple **self-attention with trainable weights**
3. Next, I implemented **causal self-attention** so that only the preceeding and current tokens are given importance
4. Finally, I implemented **Multi-head attention** using both, stacking and weight splits.

### Experiments

1. Stacking multiple attention heads leads to a slower forward pass than weight-splitting them
2. Increasing the number of attention heads reduces loss, converges faster (perhaps with a ceiling?)
3. Weight-splitting MHA is better at memory utilization than stacking multiple heads
4. Vanishing gradients if we don't scale attention scores by square root of keys matrix.
5. **Logit lens - Prove that "Reasoning" happens in the middle layers, while "Grammar" happens in the later layers.**
6. Changing learning rate and weight decay of AdamW

## Blog ideas

* What is the semantic meaning of having residual connections?
* How gradients flow across the model (including through the different transformer blocks) and how new information gets added at each layer.
* Can we *see* what's flowing through the model or what it is "thinking"?
* How does data flow through the transformer, in terms of the original input x?


### TODO

1. KV-Cache Optimization
2. Speculative Decoding
3. Model Quantization
4. Flash Attention
5. LoRA


## How to run

`uv run jupyter lab`

## Acknowledgements

Most of the core LLM implementation in this repo follows `Build a Large Language Model` by `Sebastian Raschka`.

> Raschka, Sebastian. Build A Large Language Model (From Scratch). Manning, 2024. ISBN: 978-1633437166.
