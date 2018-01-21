#imports 

from os.path import expanduser
import numpy as np
import tensorflow as tf
import sys

GLOBAL_VOCAB_SIZE = 257

def extract_data(filename):
    relative_path = "../../data/"
    data = open(expanduser(relative_path+filename),"rb")
    data_list = []
    for line in data:
        data_list.append(np.asarray(list(line)))
    return np.asarray(data_list)

def preprocess_data(data):
    newdata = []
    for row in data:
        newrow = np.zeros(row.shape)
        for i in range(len(row)):
            newrow[i]=ord(row[i])
        newdata.append(newrow)
    return np.array(newdata)

def encoder(hidden_size, input_data):
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(encoder_cell, input_data, initial_state = initial_state, dtype=tf.float32)
    return outputs, state

def decoder(hidden_size, input_data):
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    helper = tf.contrib.seq2seq.TrainingHelper(input_data, decoder_lengths)
    initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(decoder_cell, input_data, initial_state = initial_state, dtype=tf.float32)
    return outputs

def sentence_to_one_hot(sentence_numpy_array):
    one_hot_data = np.zeros((len(sentence_numpy_array),GLOBAL_VOCAB_SIZE))
    for i in range(len(sentence_numpy_array)):
        one_hot_data[i][int(sentence_numpy_array[i])]=1
    return one_hot_data

def data_to_one_hot(data):
    data_list = []
    for sentence in data:
        data_list.append(sentence_to_one_hot(sentence))
    return data_list
    # return one hot vectors for each input character

def generate_decoder_target(numpy_sequence):
    #EOS at the end
    seq_reshaped=numpy_sequence.reshape((1,numpy_sequence.shape[0],numpy_sequence.shape[1]))
    EOS = np.zeros((1,1,257))
    EOS[0,0,256] = 1
    return np.append(x,EOS,axis=1)

def generate_decoder_input(numpy_sequence):
    #EOS at the beginning
    seq_reshaped=numpy_sequence.reshape((1,numpy_sequence.shape[0],numpy_sequence.shape[1]))
    EOS = np.zeros((1,1,257))
    EOS[0,0,256] = 1
    return np.append(EOS,x,axis=1)

def generate_encoder_input(numpy_sequence):
    return numpy_sequence.reshape((1,numpy_sequence.shape[0],numpy_sequence.shape[1]))


def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    initial_input = tf.one_hot(256,257)
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input
    
    elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended

    finished = tf.reduce_all(elements_finished) # -> boolean scalar
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished, 
            input,
            state,
            output,
            loop_state)


"""
Prepare Data
"""

#extract data
train_english = extract_data("train.10k.en")
train_german = extract_data("train.10k.de")
valid_english = extract_data("valid.100.en")
valid_german = extract_data("valid.100.de")

#preprocess data
train_english_processed = preprocess_data(train_english)
train_german_processed = preprocess_data(train_german)
valid_english_processed = preprocess_data(valid_english)
valid_german_processed = preprocess_data(valid_german)

#data to one hot
one_hot_train_english = data_to_one_hot(train_english_processed)
one_hot_train_german = data_to_one_hot(train_german_processed)
one_hot_valid_english = data_to_one_hot(valid_english_processed)
one_hot_valid_german = data_to_one_hot(valid_german_processed)

#inputs and outputs

tf.reset_default_graph() 
sess = tf.InteractiveSession() 

batch_size = 1
max_sequence_length = 10
encoder_inputs = tf.placeholder(shape=(batch_size, None, GLOBAL_VOCAB_SIZE), dtype=tf.float32, name='encoder_inputs')
decoder_inputs = tf.placeholder(shape=(batch_size, None, GLOBAL_VOCAB_SIZE), dtype=tf.float32, name='decoder_inputs')
decoder_targets = tf.placeholder(shape=(batch_size, None, GLOBAL_VOCAB_SIZE), dtype=tf.float32, name='decoder_targets')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

#encoder network
hidden_size = 100
batch_size = 100
with tf.variable_scope('encoder'):
    encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    (encoder_output, encoder_final_state) = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs, dtype=tf.float32)

#decoder network
with tf.variable_scope('decoder'):
    decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    (decoder_output, decoder_final_state) = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs, initial_state = encoder_final_state, sequence_length = encoder_inputs_length, dtype=tf.float32)

decoder_logits = tf.contrib.layers.linear(decoder_output, GLOBAL_VOCAB_SIZE)
decoder_prediction = tf.argmax(decoder_logits)


#optimizer
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=decoder_targets,
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)


# start session
sess.run(tf.global_variables_initializer())

data_train = one_hot_train_english
data_labels = one_hot_train_german
epochs = 100
for e in range(epochs):
    print("epoch: ", e)
    pred = sess.run([decoder_prediction], feed_dict={encoder_inputs: generate_encoder_input(one_hot_valid_english[0])})
    print pred
    for i in range(len(data_train)):
        input_sequence = data_train[i]
        output_sequence = data_labels[i]

        sess.run([loss, train_op],
            feed_dict={
                encoder_inputs: generate_encoder_input(input_sequence),
                decoder_inputs: generate_decoder_input(output_sequence),
                decoder_targets: generate_decoder_target(output_sequence),
            })
