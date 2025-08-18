import numpy as np
import tensorflow as tf

def get_angles(pos, k, d):
    i = k // 2
    angles = pos / 10000** (2*i/d)
    return angles

def positional_encoding(positions, d):
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis], np.arange(d)[np.newaxis, :], d)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(decoder_token_ids):
    seq = 1 - tf.cast(tf.math.equal(decoder_token_ids, 0), tf.float32)
    return seq[:, tf.newaxis, :]

def create_look_ahead_mask(sequence_length):
    mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)
    return mask

def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (1. - mask) * -1e9
    attention_weights = tf.keras.activations.softmax(scaled_attention_logits, axis=-1)
    outputs = tf.matmul(attention_weights, v)
    return outputs, attention_weights

def FullyConnected(embedding_dim, fully_connected_dim):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(fully_connected_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(embedding_dim, kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(EncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim,dropout=dropout_rate)
        self.ffn = FullyConnected(
            embedding_dim=embedding_dim, fully_connected_dim=fully_connected_dim)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        self_mha_output = self.mha(x, x, x, attention_mask=mask, training=training)
        skip_x_attention = self.layernorm1(x + self_mha_output)
        ffn_output = self.ffn(skip_x_attention)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        encoder_layer_output = self.layernorm2(skip_x_attention + ffn_output)
        return encoder_layer_output

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
                 maximum_position_encoding, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)
        self.enc_layers = [EncoderLayer(embedding_dim=self.embedding_dim,
                                       num_heads=num_heads,
                                       fully_connected_dim=fully_connected_dim,
                                       dropout_rate=dropout_rate,
                                       layernorm_eps=layernorm_eps)
                        for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(tf.shape(x)[-1], dtype=tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        return x

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, fully_connected_dim,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                       key_dim=embedding_dim,
                                                       dropout=dropout_rate)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                       key_dim=embedding_dim,
                                                       dropout=dropout_rate)
        self.ffn = FullyConnected(embedding_dim=embedding_dim,
                                  fully_connected_dim=fully_connected_dim)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=layernorm_eps)
        self.dropout_ffn = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        mult_attn_out1, attn_weights_block1 = self.mha1(
            x, x, x, attention_mask=look_ahead_mask, return_attention_scores=True)
        Q1 = self.layernorm1(mult_attn_out1 + x)
        mult_attn_out2, attn_weights_block2 = self.mha2(
            Q1, enc_output, enc_output, attention_mask=padding_mask, return_attention_scores=True)
        mult_attn_out2 = self.layernorm2(mult_attn_out2 + Q1)
        ffn_output = self.ffn(mult_attn_out2)
        ffn_output = self.dropout_ffn(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + mult_attn_out2)
        return out3, attn_weights_block1, attn_weights_block2

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, 
                 target_vocab_size, maximum_position_encoding,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, self.embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embedding_dim)
        self.dec_layer = [DecoderLayer(embedding_dim=self.embedding_dim,
                                       num_heads=num_heads,
                                       fully_connected_dim=fully_connected_dim,
                                       dropout_rate=dropout_rate,
                                       layernorm_eps=layernorm_eps)
                        for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(tf.shape(x)[-1], dtype=tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layer[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1_self_att'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2_decenc_att'.format(i + 1)] = block2
        return x, attention_weights

class Transformer(tf.keras.layers.Layer):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
                 targer_vocab_size, max_position_encoding_input, max_position_encoding_target,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers=num_layers,
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               input_vocab_size=input_vocab_size,
                               maximum_position_encoding=max_position_encoding_input,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)
        self.decoder = Decoder(num_layers=num_layers,
                               embedding_dim=embedding_dim,
                               num_heads=num_heads,
                               fully_connected_dim=fully_connected_dim,
                               target_vocab_size=targer_vocab_size,
                               maximum_position_encoding=max_position_encoding_target,
                               dropout_rate=dropout_rate,
                               layernorm_eps=layernorm_eps)
        self.final_layer = tf.keras.layers.Dense(targer_vocab_size, activation='softmax')

    def call(self, input_sentence, output_sentence, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(input_sentence, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(output_sentence, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights
