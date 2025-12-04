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
