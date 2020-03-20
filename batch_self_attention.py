def multihead_attentionb(queries, 
                        keys, 
                        num_units=None, 
                        num_heads=8, 
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention", 
                        reuse=None):
    

    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        
        # Linear projections
        Q = tf.contrib.layers.fully_connected(queries, num_units) # (b,N, T_q, C)
        K = tf.contrib.layers.fully_connected(queries, num_units ) # (b,N, T_k, C)
        V = tf.contrib.layers.fully_connected(keys, num_units) # (b, N, T_k, C)
        
#         Q1 = tf.reshape(Q,(batch_size,Q.get_shape().as_list()[0],Q.get_shape().as_list()[1],num_units))
#         K1 = tf.reshape(K,(batch_size,Q.get_shape().as_list()[0],Q.get_shape().as_list()[1],num_units))        
#         V1 = tf.reshape(V,(batch_size,Q.get_shape().as_list()[0],Q.get_shape().as_list()[1],num_units))
        
        N = Q.get_shape().as_list()[1]
        TQK = Q.get_shape().as_list()[2]
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis = 3),axis =1) # (b,h*N, T_q, C/h) 
        K_ = tf.concat(tf.split(K, num_heads, axis = 3),axis =1) # (b,h*N, T_k, C/h) 
        V_ = tf.concat(tf.split(V, num_heads, axis = 3),axis =1) # (b,h*N, T_k, C/h) 

        # Multiplication
        Q_ = tf.concat(tf.split(Q_, num_heads*N, axis = 1),axis =0) # (b*h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K_, num_heads*N, axis = 1),axis =0)
        V_ = tf.concat(tf.split(V_, num_heads*N, axis = 1),axis =0)
        Q_ = tf.squeeze(Q_)
        K_ = tf.squeeze(K_)
        V_ = tf.squeeze(V_)
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # (h*N, T_q, T_k)        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
        outputs = tf.concat(tf.split(tf.expand_dims(outputs,1), num_heads*N, axis = 0),axis = 1)

                
        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1))) # (b,N,T_k)
        key_masks = tf.tile(key_masks, [1, num_heads, 1]) # (b,h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 2), [1,1, tf.shape(queries)[2],1]) # (h*N, T_q, T_k)
       # 
        paddings = tf.ones_like(outputs)*(-2**32+1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs) # (b,h*N, T_q, T_k)
          
        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0,0, :, :]) # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tf.expand_dims(tril, 0),0), [tf.shape(outputs)[0],tf.shape(outputs)[1], 1, 1]) # (b,h*N, T_q, T_k)
            paddings = tf.ones_like(masks)*(-2**32+1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # (b, h*N, T_q, T_k)
        matt    = outputs
        
        
        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1))) # (b,N, T_q)
        query_masks = tf.tile(query_masks, [1, num_heads, 1]) # (b,h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1,1, 1, tf.shape(keys)[2]]) # (h*N, T_q, T_k)
        outputs *= query_masks # broadcasting. (b, N, T_q, C)
          
        # Dropouts
        outputs = tf.contrib.layers.dropout(outputs, keep_prob=dropout_rate, is_training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.concat(tf.split(outputs, num_heads*N, axis = 1),axis =0)        
        outputs = tf.matmul(tf.squeeze(outputs), V_) # (b*h*N, T_q, C/h)
        outputs = tf.concat(tf.split(tf.expand_dims(outputs,1), num_heads*N, axis = 0),axis =1) # (b,h*N, T_q, C/h)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads,axis = 1),axis =3) # (b,N, T_q, C)
              
        # Residual connection
        outputs += queries              
              
        # Normalize
        for i in range(0,batch_size):            
            outputsx = normalize(outputs[i,:,:,:]) # (N, T_q, C)
            if i == 0:
                outputsxx = tf.reshape(outputsx,(1,N,TQK,num_units))
            else:
                temp_outputsxx = tf.reshape(outputsx,(1,N,TQK,num_units))
                outputsxx = tf.concat([outputsxx, temp_outputsxx],axis =0)

    return outputsxx, matt