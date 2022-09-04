"""
@author : Arnaud HINCELIN
@date : 27/04/2022
"""

import os
import tensorflow as tf
import numpy as np

import colorama
from colorama import Fore
from colorama import Style
from tf_agents.utils import object_identity

import tensorflow_datasets as tfds
import tensorflow_text

MAX_TOKENS = 200

###-----------------------------------------------------------------###
###                              MASKING                            ###
###-----------------------------------------------------------------###

"""
### PADDING MASK ###
-> General idea : 
Mask all the pad tokens in the batch of sequence. 
It ensures that the model does not treat padding as the input.
-> The mask indicates where pad value 0 is present: it outputs a 1 at those locations, \
    and a 0 otherwise.
So, from batched input seq, will return same array but with 1 when it is a 0, and 0 otherwise

Ex : 
[[1,2,0,4], [1,0,7,8], [1,2,0,0]] #(3, 4)
->
[ [[[0. 0. 1. 0.]]], [[[0. 1. 0. 0.]]], [[[0. 0. 1. 1.]]] ] #(3, 1, 1, 4)
Add dimension 2 and 3 to add the padding
"""
def create_padding_mask(seq):
    inter = tf.math.equal(seq, 0) #bool array
    seq = tf.cast( inter , tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


"""
### LOOK-AHEAD MASK ###
-> General idea : 
The look-ahead mask is used to mask the future tokens in a sequence. 
In other words, the mask indicates which entries should not be used.

-> 
This means that to predict the third token, only the first and second token will be used.

Ex : 
size=4
-> 
[[0. 1. 1. 1.]  #predict 1st token
 [0. 0. 1. 1.]  #predict 2nd token
 [0. 0. 0. 1.]  #predict 3th token
 [0. 0. 0. 0.]] #predict 4th token
"""
def create_look_ahead_mask(size):
    matrix_full_one = tf.ones( shape=(size, size) )
    # band_part() : Copy a tensor setting everything outside a central band in each innermost matrix to zero.
    mask = tf.linalg.band_part( 
        input=matrix_full_one, 
        num_lower=-1, 
        num_upper=0
        )
    mask = 1 - mask

    return mask  # (seq_len, seq_len)


"""
### MASKING within Transformer ###
MASK in attention (padding or look-ahead): 
    scaled_attention_logits += (mask * -1e9)
    softmax( scaled_attention_logits )

Be careful : We create mask before word embedding ! 
So seq input and target are array of integers (id of tokens) ! 

-->> MASK Pipeline : 
1 - Encoder
- Multi-head Layer : PADDING on the input sequence
input_seq -> |PADDING| -> input_padding_mask

2 - Decoder 
- Multi-head Layer 1 : PADDING and LOOK-AHEAD MASKS on the target sequence
target_seq -> |PADDING| -> target_padding_mask -> |LOOK-AHEAD| -> target_lookahead_padding_mask

- Multi-head Layer 2 : PADDING MASKS on the input sequence
input_seq -> |PADDING| -> input_padding_mask

-->> Ex : 
input_seq= [[1,8,3,56,0]] -> 4 words 
target_seq= [[1,23,4,0,0]] -> 3 words

input_padding_mask : 
[[[[0. 0. 0. 0. 1.]]]] #(1, 1, 1, 5) = (batch_size, 1, 1, seq_len)

target_lookahead_padding_mask : 
[[[[0. 1. 1. 1. 1.]
   [0. 0. 1. 1. 1.]
   [0. 0. 0. 1. 1.]
   [0. 0. 0. 1. 1.]
   [0. 0. 0. 1. 1.]]]] #(1, 1, 5, 5) = (batch_size, 1, seq_len, seq_len)

def create_masks(input_seq, target_seq):
    # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
    input_padding_mask = create_padding_mask(input_seq)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    seq_target_len = shape(target_seq)[1]
    target_look_ahead_mask = create_look_ahead_mask( seq_target_len ) #(seq_target_len, seq_target_len)
    target_padding_mask = create_padding_mask(target_seq) #(batch_size, 1, 1, seq_len)
    target_lookahead_padding_mask = maximum(target_padding_mask, target_look_ahead_mask) #(seq_target_len, seq_target_len)

    return input_padding_mask, target_lookahead_padding_mask

"""

###-----------------------------------------------------------------###
###                           POSITIONNAL                           ###
###-----------------------------------------------------------------###

"""
Attention Layer sees input as a set of vectors, with no suequential order.
Positionnal encoding is added to give the model some info about relative position of tokens.

-->> INPUT_WORD_VECTOR = WORD_EMBEDDING_VECTOR + WORD_POSITIONNAL_VECTOR
- WORD_EMBEDDING_VECTOR represents a token in space where tokens with similar meaning \
    will be closer to each other
- WORD_POSITIONNAL_VECTOR encodes the relative position of tokens in a sentence

- INPUT_WORD_VECTOR : tokens will be closer to each other based on the similarity \
    of their meaning and their position in the sentence, in the d-dimensional space

pos : position of token within the sentence
i : position of value within embedded vector
PE(pos, 2i) = sin( pos / 10000^(2i/d_model) )
PE(pos, 2i+1) = sin( pos / 10000^( (2i+1)/d_model ) )

"""
def get_angles(pos, i, d_model):
    exp = (2 * (i//2)) / np.float32(d_model)
    angle_rates = np.power( 10000, exp )
    angle_rates = pos / angle_rates
    return angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], #(position, 1)
        np.arange(d_model)[np.newaxis, :], #(1, d_model)
        d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[ np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

###-----------------------------------------------------------------###
###                              ATTENTION                          ###
###-----------------------------------------------------------------###
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Attention(Q,K,V) = Softmax[ (Q*K.T)/sqrt(N) ]*V -->> Autant que de nb de mots de la seq

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast( tf.shape(k)[-1], tf.float32 )
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) 
     #(batch_size, num_heads, seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  ##(batch_size, num_heads, seq_len_q, depth)

    return output, attention_weights


###-----------------------------------------------------------------###
###                            FEED-FORWARD                         ###
###-----------------------------------------------------------------###
def point_wise_feed_forward_network(d_model, dff):
    mini_model = tf.keras.Sequential()
    mini_model.add( tf.keras.layers.Dense(dff, activation='relu') )  # (batch_size, seq_len, dff)
    mini_model.add( tf.keras.layers.Dense(d_model) )  # (batch_size, seq_len, d_model)
    return mini_model


###-----------------------------------------------------------------###
###                       CLASS MULTI-HEAD                          ###
###-----------------------------------------------------------------###
class MultiHeadAttention_layer(tf.keras.layers.Layer):
    #For simplicity/efficiency the code below implements this using a single 
    # dense layer with num_heads times as many outputs. 
    # The output is rearranged to a shape of (batch, num_heads, ...) before applying the attention function.
    def __init__(self,*, d_model, num_heads):
        super(MultiHeadAttention_layer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads #512/8 = 64

        # Each Head has its own Denses Layers (x3) to create Q,K, and V
        ## So normally, we have 3 x nb_heads Denses layers
        # But we packed all Denses into 3 single big Denses Layers, one for Q, K and V
        # 64 units for each Head (x8), so we have Denses of 512 units, 
        # and each part of the weight matrix is refered to a Head
        self.layer_Q = tf.keras.layers.Dense(d_model) # (512 units)
        self.Layer_K = tf.keras.layers.Dense(d_model) # (512)
        self.Layer_V = tf.keras.layers.Dense(d_model) 

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        #Split the last dimension into (num_heads, depth).
        #Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        # input shape : (batch_size, seq_len, d_model) = (1, 128, 512)
        # output shape : (batch_size, num_heads, seq_len_q, depth) = (1, 8, 128, 64)

        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  #(1, 128, 8, 64)
        return tf.transpose(x, perm=[0, 2, 1, 3]) #transpose with perm 2 and 1 : (1, 8, 128, 64)

    def call(self, q, k, v, mask):
        batch_size = tf.shape(v)[0]

        queries = self.layer_Q( q )  # (batch_size, seq_len, d_model) = (1, 128, 512 )
        keys = self.Layer_K( k )  # (batch_size, seq_len, d_model)
        values = self.Layer_V( v )  # (batch_size, seq_len, d_model)

        queries = self.split_heads(queries, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        keys = self.split_heads(keys, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        values = self.split_heads(values, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth) = (1, 8, 128, 64)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k) = (1, 8, 128, 128)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q=queries, k=keys, v=values, 
            mask=mask)

        scaled_attention = tf.transpose(
            scaled_attention, 
            perm=[0, 2, 1, 3]
            ) #(batch_size, seq_len_q, num_heads, depth)=(1, 128, 8, 64)

        concat_attention = tf.reshape(
            tensor=scaled_attention,
            shape=(batch_size, -1, self.d_model)
            )  # (batch_size, seq_len_q, d_model) = (1, 128, 512)

        output = self.dense(concat_attention) # (batch_size, seq_len_q, d_model) = (1, 128, 512)

        return output, attention_weights


###-----------------------------------------------------------------###
###                       CLASS SINGLE ENCODER                      ###
###-----------------------------------------------------------------###
class EncoderLayer( tf.keras.layers.Layer ):
    def __init__(self,*, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention_layer(d_model=d_model, num_heads=num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        # input x : seq of WE vectors : # (batch_size, input_seq_len, d_model)
        #-> Compute attention with mha Layer
        attn_output, attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        #-> Dropout Layer
        attn_output = self.dropout1(attn_output, training=training)
        # -> Residual connection + Norm Layer
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        #-> FFNN Layer
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        #-> Dropout Layer
        ffn_output = self.dropout2(ffn_output, training=training)
        # -> Residual connection + Norm Layer
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attn_weights

###-----------------------------------------------------------------###
###                       CLASS SINGLE DECODER                      ###
###-----------------------------------------------------------------###
class DecoderLayer( tf.keras.layers.Layer ):
    def __init__(self,*, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention_layer(d_model=d_model, num_heads=num_heads)
        self.mha2 = MultiHeadAttention_layer(d_model=d_model, num_heads=num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
            look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        #-> Compute MASKED OUTPUT-OUTPUT attention with mha1 Layer
        attn1, attn_weights_block1 = self.mha1(
            q=x, k=x, v=x, mask=look_ahead_mask
            )  
        # attn1 : (batch_size, target_seq_len, d_model)

        #-> Dropout Layer
        attn1 = self.dropout1(attn1, training=training)
        # -> Residual connection + Norm Layer
        out1 = self.layernorm1(attn1 + x)

        #-> Compute ENCODER-DECODER attention with mha2 Layer
        attn2, attn_weights_block2 = self.mha2(
            q=out1, k=enc_output, v=enc_output, mask=padding_mask
            )  
        # attn2 :(batch_size, target_seq_len, d_model)

        #-> Dropout Layer
        attn2 = self.dropout2(attn2, training=training)
        # -> Residual connection + Norm Layer
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        #-> FFNN Layer
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        #-> Dropout Layer
        ffn_output = self.dropout3(ffn_output, training=training)
        # -> Residual connection + Norm Layer
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

###-----------------------------------------------------------------###
###                           CLASS ENCODER                         ###
###-----------------------------------------------------------------###
class Encoder( tf.keras.layers.Layer ):
    def __init__(self,*, num_layers, d_model, num_heads, dff, 
    input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)

        self.pos_encoding = positional_encoding(MAX_TOKENS, self.d_model) #(1, len_seq_input, d_model)

        self.enc_layers = [
            EncoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        #input : (batch_size, input_seq_len)
        seq_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]
        attention_weights = {}

        # -> Embedding Layer
        x_we = self.embedding(x)  #float32 (batch_size, input_seq_len, d_model)
                
        # -> positionnal enconding
        x_we *= tf.math.sqrt( tf.cast(self.d_model, tf.float32) )
        x_we += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x_we, training=training)

        for i in range(self.num_layers):
            x, attn_weights = self.enc_layers[i](x, training, mask)
            attention_weights[f'encoder_layer{i+1}'] = attn_weights

        return x, attention_weights  # (batch_size, input_seq_len, d_model)


###-----------------------------------------------------------------###
###                           CLASS DECODER                         ###
###-----------------------------------------------------------------###
class Decoder( tf.keras.layers.Layer ):
    def __init__(self,*, num_layers, d_model, num_heads, dff, 
    target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, self.d_model)

        self.pos_encoding = positional_encoding(MAX_TOKENS, self.d_model) #(1, len_seq_input, d_model)

        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads, dff=dff, rate=rate)
            for _ in range(num_layers)]
            
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]
        attention_weights = {}

        # -> Embedding Layer
        x_we = self.embedding(x)  #float32 (batch_size, input_seq_len, d_model)

        # -> positionnal enconding
        x_we *= tf.math.sqrt( tf.cast(self.d_model, tf.float32))
        x_we += self.pos_encoding[:, :seq_len, :]

        # -> dropout
        x = self.dropout(x_we, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, 
                training,
                look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

###-----------------------------------------------------------------###
###                           CLASS TRANSFORMER                     ###
###-----------------------------------------------------------------###
class Transformer( tf.keras.Model ):
    def __init__(self,*, num_layers, d_model, num_heads, dff, input_vocab_size,
                target_vocab_size, rate=0.1, name, nb_training_sample):
        super().__init__()
        self.heads = num_heads
        self.feed_forward = dff
        self.name_transfo = name
        self.nb_training_sample = nb_training_sample

        self.encoder = Encoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, 
            dff=dff, input_vocab_size=input_vocab_size, rate=rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model, num_heads=num_heads, 
            dff=dff, target_vocab_size=target_vocab_size, rate=rate)   

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        input_padding_mask, target_lookahead_padding_mask = self.create_masks(inp, tar)

        # -> Encoder Layer (xN Encoders)
        encoder_output, encoder_attention_weights = self.encoder(
            x=inp, 
            training=training, 
            mask=input_padding_mask
            )  # encoder_output : (batch_size, inp_seq_len, d_model)

        # -> Decoder Layer (xN Decoders)
        dec_output, decoder_attention_weights = self.decoder(
            x=tar, 
            enc_output=encoder_output, 
            training=training, 
            look_ahead_mask=target_lookahead_padding_mask, 
            padding_mask=input_padding_mask
            )
            # dec_output.shape == (batch_size, target_seq_len, d_model)

        final_output = self.final_layer(dec_output)  #(batch_size, tar_seq_len, target_vocab_size)

        return final_output, encoder_attention_weights, decoder_attention_weights

    def create_masks(self, input_seq, target_seq):
        # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
        input_padding_mask = create_padding_mask(input_seq)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        seq_target_len = tf.shape(target_seq)[1]
        target_look_ahead_mask = create_look_ahead_mask( seq_target_len )
        target_padding_mask = create_padding_mask(target_seq)
        target_lookahead_padding_mask = tf.maximum(target_padding_mask, target_look_ahead_mask)

        return input_padding_mask, target_lookahead_padding_mask

    def get_config(self):
        return {
            "layers":self.encoder.num_layers,
            "nb_heads":self.heads,
            "dim_model":self.encoder.d_model,
            "dim_feed_forward":self.feed_forward,
            "nb_traininig_sample":self.nb_training_sample,
            "name":self.name_transfo
            }





def load_saved_transformer_model(path):
    assert isinstance(path, str)
    print( Style.BRIGHT+Fore.RED+ f"\n>>>>>>>>>> loading the model... <<<<<<<<<<" )
    print( Fore.YELLOW+f"Model path : {path} \n" + Style.RESET_ALL )
    
    model_transfo = tf.keras.models.load_model( path )
    # Check architecture
    print( Fore.RED+ ">>>>> ARCHITECTURE OF LOADED MODEL : " + Style.RESET_ALL )
    model_transfo.summary()

    w = model_transfo.trainable_weights
    nb_params = int( sum( np.prod(p.shape.as_list()) for p in object_identity.ObjectIdentitySet(w)))
    size_bytes = nb_params*4

    print( Style.BRIGHT+Fore.CYAN+f"SIZE OF MODEL : {size_bytes} bytes -->> {size_bytes*9.31e-10:.3f} GB \n"+Style.RESET_ALL )

    # Check config
    config_dict = model_transfo.get_config()
    print( Fore.RED+ ">>>>> CONFIGURATION OF LOADED MODEL : " + Style.RESET_ALL )
    for key in config_dict.keys():
        print( f"{key:20}: {config_dict.get(key)}" )
    print("\n")
    return model_transfo

###-----------------------------------------------------------------###
###                     CLASS TRANSLATOR - INFERENCE                ###
###-----------------------------------------------------------------###

class Translator( tf.Module ):
    def __init__(self, tokizer_input, tokizer_out, transformer):
        self.tok_in = tokizer_input
        self.tok_out = tokizer_out
        self.transformer = transformer

    def __call__(self, sentence, max_length=MAX_TOKENS):
        # input sentence is portuguese, hence adding the start and end token
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = self.tok_in.tokenize(sentence)
        sentence = sentence.to_tensor() #(batch_size=1, input_seq_len)

        encoder_input = sentence

        # As the output language is english, initialize the output with the
        # english start token.
        start_end = self.tok_out.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        # `tf.TensorArray` is required here (instead of a python list) so that the
        # dynamic-loop can be traced by `tf.function`.
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)

        
        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())

            predictions, _ , _ = self.transformer([encoder_input, output], training=False)

            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)

            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output_array = output_array.write(i+1, predicted_id[0])

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())
        # output.shape (1, tokens)

        text_predicted = self.tok_out.detokenize(output)[0]  # shape: ()
        text_predicted_tokens = self.tok_out.lookup(output)[0]

        # `tf.function` prevents us from using the attention_weights that were
        # calculated on the last iteration of the loop. So recalculate them outside
        # the loop.
        _ , encoder_attention, decoder_attention = self.transformer([encoder_input, output[:,:-1]], training=False)
        output_tokens = output

        return text_predicted, text_predicted_tokens, encoder_attention, decoder_attention





