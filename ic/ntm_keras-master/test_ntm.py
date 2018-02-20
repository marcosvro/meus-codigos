from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, SGD
from ntm import controller_input_output_shape as controller_shape
from ntm import NeuralTuringMachine as NTM
from keras.callbacks import TensorBoard, ModelCheckpoint, TerminateOnNaN
import numpy as np


def get_sample(batch_size=128, in_bits=10, out_bits=8, max_size=20, min_size=1, samples=1000):
    # in order to be a generator, we start with an endless loop:
    while True:
        # generate samples with random length.
        # there a two flags, one for the beginning of the sequence 
        # (only second to last bit is one)
        # and one for the end of the sequence (only last bit is one)
        # every other time those are always zero.
        # therefore the length of the generated sample is:
        # 1 + actual_sequence_length + 1 + actual_sequence_length
        
        # make flags
        begin_flag = np.zeros((1, in_bits))
        begin_flag[0, in_bits-2] = 1
        end_flag = np.zeros((1, in_bits))
        end_flag[0, in_bits-1] = 1
    
        # initialize arrays: for processing, every sequence must be of the same length.
        # We pad with zeros.
        temporal_length = max_size*2 + 2
        # "Nothing" on our band is represented by 0.5 to prevent immense bias towards 0 or 1.
        inp = np.ones((batch_size, temporal_length, in_bits))*0.5
        out = np.ones((batch_size, temporal_length, out_bits))*0.5
        # sample weights: in order to make recalling the sequence much more important than having everything set to 0
        # before and after, we construct a weights vector with 1 where the sequence should be recalled, and small values
        # anywhere else.
        sw  = np.ones((batch_size, temporal_length))*0.0001
    
        # make actual sequence
        for i in range(batch_size):
            ts = np.random.randint(low=min_size, high=max_size+1)
            actual_sequence = np.random.uniform(size=(ts, out_bits)) > 0.5
            output_sequence = np.concatenate((np.ones((ts+2, out_bits))*0.5, actual_sequence), axis=0)
    
            # pad with zeros where only the flags should be one
            padded_sequence = np.concatenate((actual_sequence, np.zeros((ts, 2))), axis=1)
            input_sequence = np.concatenate((begin_flag, padded_sequence, end_flag), axis=0)
            
    
            # this embedds them, padding with the neutral value 0.5 automatically
            inp[i, :input_sequence.shape[0]] = input_sequence
            out[i, :output_sequence.shape[0]] = output_sequence
            sw[i, ts+2 : ts+2+ts] = 1
        yield inp, out, sw



def test_model(model, sequence_length=None, verboose=False):
    input_dim = model.input_dim
    output_dim = model.output_dim
    batch_size = model.batch_size

    I, V, sw = next(get_sample(batch_size=batch_size, in_bits=input_dim, out_bits=output_dim, max_size=sequence_length, min_size=sequence_length))
    Y = np.asarray(model.predict(I, batch_size=batch_size))

    if not np.isnan(Y.sum()): #checks for a NaN anywhere
        Y = (Y > 0.5).astype('float64')
        x = V[:, -sequence_length:, :] == Y[:, -sequence_length:, :]
        acc = x.mean() * 100
        if verboose:
            print("the overall accuracy for sequence_length {0} was: {1}".format(sequence_length, x.mean()))
            print("per bit")
            print(x.mean(axis=(0,1)))
            print("per timeslot")
            print(x.mean(axis=(0,2)))
    else:
        ntm = model.layers[0]
        #weights = ntm.get_weights()
        #import pudb; pu.db
        acc = 0
    return acc





# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
output_dim = 8
input_dim = output_dim + 2  # this is the actual input dim of the network, that includes two dims for flags
batch_size = 10
read_heads = 1
write_heads = 1
n_slots = 128
m_depth = 20
shift_range = 3


lr=9e-4
clipnorm = 10
epochs = 10
num_samples = 1000
otm = Adam(lr=lr, clipnorm=clipnorm)

controller_input_dim, controller_output_dim = controller_shape(input_dim, output_dim, m_depth, n_slots, shift_range, read_heads, write_heads)

controller = Sequential()
controller.name = "LSTM controller"
controller.add(LSTM(units=controller_output_dim,
                    kernel_initializer='random_normal', 
                    bias_initializer='random_normal',
                    activation='linear',
                    stateful=True,
                    #implementation=2,   # best for gpu. other ones also might not work.
                    batch_input_shape=(batch_size, None, controller_input_dim)))

controller.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['binary_accuracy'], sample_weight_mode="temporal")

model = Sequential()
model.name = "NTM with " + controller.name
model.batch_size = batch_size
model.input_dim = input_dim
model.output_dim = output_dim
model.add(NTM(output_dim, n_slots=n_slots, m_depth=m_depth, shift_range=shift_range,
              controller_model=controller,
              activation="sigmoid",
              read_heads = read_heads,
              write_heads = write_heads,
              return_sequences=True,
              input_shape=(None, input_dim),
              batch_size = batch_size))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['binary_accuracy'], sample_weight_mode="temporal")


print("model built, starting the copy experiment")

log_path = "logs"
tensorboard =   TensorBoard(log_dir=log_path,
                            write_graph=False, #This eats a lot of space. Enable with caution!
                            #histogram_freq = 1,
                            write_images=True,
                            batch_size = model.batch_size,
                            write_grads=True)
model_saver =  ModelCheckpoint(log_path + "\model.ckpt.{epoch:04d}.hdf5", monitor='loss', period=1)
callbacks = [tensorboard, TerminateOnNaN(), model_saver]

sample_generator = get_sample(batch_size=batch_size, in_bits=input_dim, out_bits=output_dim, max_size=20, min_size=5, samples=num_samples)


model.fit_generator(sample_generator, steps_per_epoch=100, epochs=1, verbose=1, callbacks=callbacks)

for i in [10, 20, 40, 80]:
        acc = test_model(model, sequence_length=i)
        print("the accuracy for length {0} was: {1}%".format(i,acc))




