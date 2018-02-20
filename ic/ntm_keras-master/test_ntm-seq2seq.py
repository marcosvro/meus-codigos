# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:49:26 2018

@author: marcos
"""
from keras.layers import Input, Dense, LSTM
from ntm import NeuralTuringMachine as NTM
from ntm import controller_input_output_shape as controller_shape
from keras.models import Model


num_encoder_tokens = 20
num_decoder_tokens = 22
layer_dim = 256
bs = 32
ep = 10
num_samples = 10000

m_depth = 32
n_slots = 64
shift_range = 3
read_heads = 1
write_heads = 1


controller_input_dim, controller_output_dim = controller_shape(num_encoder_tokens, 
                                                               layer_dim, 
                                                               m_depth, 
                                                               n_slots, 
                                                               shift_range, 
                                                               read_heads, 
                                                               write_heads)



encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = NTM(layer_dim,
              n_slots=n_slots,
              m_depth=m_depth,
              shift_range=shift_range,
              controller_model=None,
              activation="sigmoid",
              read_heads = read_heads,
              write_heads = write_heads,
              return_sequences=True,
              return_state=True)
saidas = encoder(encoder_inputs)

print(saidas[1])










