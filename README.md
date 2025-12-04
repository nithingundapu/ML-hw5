# ML-hw5
#Nithin Gundapu
#700772575

# question 1
### Scaled Dot-Product Attention (Explanation)

This code implements the core attention mechanism used in Transformer models.

1. **softmax(x)**  
   A numerically stable softmax function. It subtracts the max value to prevent overflow, then normalizes the exponentiated values into probabilities.

2. **scaled_dot_product_attention(Q, K, V)**  
   Computes attention in three steps:
   - **Scores = Q · Kᵀ / √dₖ**  
     Measures similarity between queries and keys, scaled to keep values stable.
   - **Softmax(scores)**  
     Converts scores into attention weights that sum to 1 across each sequence.
   - **Context = weights · V**  
     Produces a weighted combination of value vectors.

3. **Test section**  
   Random Q, K, and V are created and passed through the attention function.  
   Output shapes confirm correctness:  
   - Attention weights → `(batch, seq_len, seq_len)`  
   - Context vectors → `(batch, seq_len, d_v)`


#question2

#  Self-Attention & Transformer Encoder (PyTorch)

This module implements the core components of a Transformer encoder layer.

# MultiHeadSelfAttention
The `MultiHeadSelfAttention` class performs:
1. **Linear projections**  
   Input `x` is projected into Query (Q), Key (K), and Value (V) matrices.
2. **Head splitting**  
   The vectors are reshaped into multiple attention heads for parallel processing.
3. **Scaled dot-product attention**  
   \[
   \text{scores} = \frac{QK^T}{\sqrt{d_{\text{head}}}}
   \]
   Softmax converts scores into attention weights.
4. **Combine heads**  
   Outputs from all heads are merged and passed through a final linear layer.

#  TransformerEncoder
The encoder block includes:
- **Multi-head self-attention**
- **Residual connections**
- **Layer normalization**
- **Feed-forward network (FFN)**  
A two-layer MLP expands and projects features back to `d_model`.

#  Test Output
Running the test script verifies correct shape flow:
- Encoder output: **(batch, seq_len, d_model)** → `(32, 10, 128)`
- Attention weights: **(batch, heads, seq, seq)** → `(32, 8, 10, 10)`

This confirms that each token attends to every other token across all attention heads.

