# Multi-head-Self-Attention-with-extra-batch
Perform Self-attention with several input batches
# Usage:    
    Args:
      queries: A 4d tensor with shape of [B, N, T_q, C_q].
      keys: A 4d tensor with shape of [B, N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 4d tensor with shape of (B, N, T_q, C)  

