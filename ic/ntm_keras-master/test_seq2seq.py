# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:46:38 2018

@author: marcos
"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model


def get_samples(tokkens = 10, samples=1000, min_range=1, max_range=20, verbose=0):
    X_encoder = np.zeros((samples, max_range, tokkens), dtype='float32')
    X_decoder = np.zeros((samples, max_range+2, tokkens+2), dtype='float32')
    Y = np.zeros((samples, max_range+2, tokkens+2), dtype='float32')
    
    for i in range(samples):
        sequence_lenght = np.random.randint(low=min_range, high=max_range+1)
        
        input_seq = np.random.randint(tokkens, size=sequence_lenght)
        if verbose:
            print("Input ", i, " - ", input_seq)
        output_seq = np.array([tokkens]+np.sort(input_seq).tolist()+[tokkens+1], dtype='float32')
        target_seq = np.array(np.sort(input_seq).tolist()+[tokkens+1], dtype='float32')
        
        input_seq = to_categorical(input_seq, tokkens)
        output_seq = to_categorical(output_seq, tokkens+2)
        target_seq = to_categorical(target_seq, tokkens+2)
        
        X_encoder[i,:sequence_lenght,:] = input_seq
        X_decoder[i,:sequence_lenght+2,:] = output_seq
        Y[i,:sequence_lenght+1,:] = target_seq
    
    return X_encoder, X_decoder, Y

def decode_sequence(input_seq, max_sequence_len=20):
    states_value = encoder_model.predict(input_seq)
    
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    
    target_seq[0, 0, num_decoder_tokens-2] = 1.
    
    stop = False
    decoded_sequence = []
    while not stop:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        
        sampled_token = np.argmax(output_tokens[0, -1, :])
        decoded_sequence.append(sampled_token)
        
        if sampled_token == num_decoder_tokens-1 or len(decoded_sequence) > max_sequence_len:
                stop = True
        
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token] = 1
        
        states_value = [h, c]
        
    return decoded_sequence
        

num_encoder_tokens = 20
num_decoder_tokens = 22
layer_dim = 256
bs = 32
ep = 10
num_samples = 10000

'''
encoder_input_data, decoder_input_data, decoder_target_data = get_samples(tokkens=num_encoder_tokens,
                                                                          samples=num_samples,
                                                                          max_range=20,
                                                                          verbose=1)

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(layer_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]


decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(layer_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _= decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

checkpoint = ModelCheckpoint('seq2seq_ordenador.h5', monitor='val_loss', save_best_only=True, mode='auto', verbose=0)

model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = bs, epochs=ep, callbacks=[checkpoint])


encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(layer_dim,))
decoder_state_input_c = Input(shape=(layer_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model( [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)



#save models
encoder_model.save('encoder.h5')
decoder_model.save('decoder.h5')

'''

#load model
encoder_model = load_model('encoder.h5')
decoder_model = load_model('decoder.h5')

encoder_input_data, decoder_input_data, decoder_target_data = get_samples(tokkens=num_encoder_tokens,
                                                                          samples=3,
                                                                          max_range=20,
                                                                          verbose=1)

print("DECODING..")
for i,seq in enumerate(encoder_input_data):
    seq = seq.reshape(1, seq.shape[0], seq.shape[1])
    #print(seq.shape)
    decoded = decode_sequence(seq)
    print ("Output ", i, " - ", decoded)



