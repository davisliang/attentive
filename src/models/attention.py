import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn, Block

def one_hots(numerical_list, vocab_size):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result

def textify(embedding):
    result = ""
    indices = nd.argmax(embedding, axis=1).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result

def load_time_machine(seq_length=64, batch_size=1):
    # loading dataset
    path = "../../data/timemachine.txt"
    with open(path) as f:
        time_machine = f.read()
    time_machine = time_machine[:-38083] #hardcoded to remove crap
    character_dict, vocab_size = get_char_dict(time_machine)
    
    time_numerical = [character_dict[char] for char in time_machine]
    # -1 here so we have enough characters for labels later
    num_samples = (len(time_numerical) - 1) // seq_length
    dataset = one_hots(time_numerical[:seq_length*num_samples],vocab_size).reshape((num_samples, seq_length, vocab_size))
    num_batches = len(dataset) // batch_size
    train_data = dataset[:num_batches*batch_size].reshape((batch_size, num_batches, seq_length, vocab_size))
    
    # swap batch_size and seq_length axis to make later access easier
    train_data = nd.swapaxes(train_data, 0, 1)
    train_data = nd.swapaxes(train_data, 1, 2)
    print('Shape of data set: ', train_data.shape)
    
    labels = one_hots(time_numerical[1:seq_length*num_samples+1], vocab_size)
    train_label = labels.reshape((batch_size, num_batches, seq_length, vocab_size))
    train_label = nd.swapaxes(train_label, 0, 1)
    train_label = nd.swapaxes(train_label, 1, 2)
    print('Shape of label set: ', train_label.shape)
    
    return train_data, train_label
    

def get_char_dict(data):
    # get character dictionary
    character_list = list(set(data))
    vocab_size = len(character_list)
    # get the character dictionary
    character_dict = {}
    for e, char in enumerate(character_list):
        character_dict[char] = e
    return character_dict, vocab_size

def rnn_helper(num_hidden, vocab_size): 
    num_inputs = vocab_size
    num_outputs = vocab_size
    Wxh = nd.random_normal(shape=(num_inputs,num_hidden), ctx=ctx) * .01
    Whh = nd.random_normal(shape=(num_hidden,num_hidden), ctx=ctx) * .01
    bh = nd.random_normal(shape=num_hidden, ctx=ctx) * .01
    Why = nd.random_normal(shape=(num_hidden,num_outputs), ctx=ctx) * .01
    by = nd.random_normal(shape=num_outputs, ctx=ctx) * .01
    params = [Wxh, Whh, bh, Why, by]

    for param in params:
        param.attach_grad()
    return params
 
def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear))
    partition = nd.nansum(exp, axis=0, exclude=True)
    return exp / partition

def encoder(steps, input_data, num_hidden, vocab_size, state, params):
    Wxh, Whh, bh, Why, by = params
    outputs = []
    h = state
    for i in range(input_data.shape[0]):
        h_linear = nd.dot(input_data[i], Wxh) + nd.dot(h, Whh) + bh
        h = nd.tanh(h_linear)
        yhat_linear = nd.dot(h, Why) + by
        yhat = softmax(yhat_linear) 
        outputs.append(nd.expand_dims(yhat[0],axis=1))
    return (outputs, h)

def attention(decoder_hidden_t, encoder_output):
    if(decoder_hidden_t.shape[1] != encoder_output.shape[0]):
        encoder_output = encoder_output.T
    return nd.dot(softmax(nd.dot(decoder_hidden_t, encoder_output)) , encoder_output.T)
 
def decoder(steps, encoder_outputs, state, num_hidden, vocab_size, params):
    Wxh, Whh, bh, Why, by = params
    outputs = []
    h = state
    for i in range(steps):
        h=nd.reshape(h,(1,h.size))
        yhat = softmax(nd.dot(nd.tanh(nd.dot(attention(h, encoder_outputs), Wxh) + nd.dot(h, Whh) + bh), Why) + by) 
        outputs.append(yhat[0])
    return (outputs, h)    
    
def SGD(params, lr):    
    for param in params:
        param[:] = param - lr * param.grad
        

def cross_entropy(yhat, y):
    return - nd.mean(nd.sum(y * nd.log(yhat), axis=0, exclude=True))


def average_ce_loss(outputs, labels):
    assert(len(outputs) == len(labels))
    total_loss = 0.
    for (output, label) in zip(outputs,labels):
        total_loss = total_loss + cross_entropy(output, label)
    return total_loss / len(outputs)

class decoder_layer(Block):
    def __init__(self, steps, num_hidden, vocab_size):
        super(encoder_layer, self).__init__(**kwargs)
        with self.name_scope():
            # layer meta information
            self.steps = steps
            self.num_hidden = num_hidden
            self.vocab_size = vocab_size
            self.num_inputs = vocab_size
            self.num_outputs = vocab_size
            
            # initialize layer RNN parameters
            self.Wxh = self.params.get('d_Wxh', shape=(self.num_inputs,num_hidden), init=mx.init.Xavier(magnitude=2.24))
            self.Whh = self.params.get('d_Whh', shape=(num_hidden,num_hidden), init=mx.init.Xavier(magnitude=2.24))
            self.bh = self.params.get('d_bh', shape=num_hidden)
            self.Why = self.params.get('d_Why', shape=(num_hidden,self.num_outputs), init=mx.init.Xavier(magnitude=2.24))
            self.by = self.params.get('d_by', shape=self.num_outputs)
    def forward(self,input_data,hidden_state):
        with input_data.context:
            outputs = []
            h=state
            for i in range(input_data.shape[0]):
                h_linear = nd.dot(attention(input_data[i]), Wxh) + nd.dot(h, Whh) + bh
                h = nd.tanh(h_linear)
                yhat_linear = nd.dot(h, Why) + by
                yhat = softmax(yhat_linear) 
                outputs.append(nd.expand_dims(yhat[0],axis=1))
            return (outputs, h)

class encoder_layer(Block):
    def __init__(self, steps, num_hidden, vocab_size, **kwargs):
        super(encoder_layer, self).__init__(**kwargs)
        with self.name_scope():
            # layer meta information
            self.steps = steps
            self.num_hidden = num_hidden
            self.vocab_size = vocab_size
            self.num_inputs = vocab_size
            self.num_outputs = vocab_size
            
            # initialize layer RNN parameters
            self.Wxh = self.params.get('e_Wxh', shape=(self.num_inputs,num_hidden), init=mx.init.Xavier(magnitude=2.24))
            self.Whh = self.params.get('e_Whh', shape=(num_hidden,num_hidden), init=mx.init.Xavier(magnitude=2.24))
            self.bh = self.params.get('e_bh', shape=num_hidden)
            self.Why = self.params.get('e_Why', shape=(num_hidden,self.num_outputs), init=mx.init.Xavier(magnitude=2.24))
            self.by = self.params.get('e_by', shape=self.num_outputs)
    def forward(self,input_data, hidden_state):
        with input_data.context:
            outputs = []
            h=state
            for i in range(input_data.shape[0]):
                h_linear = nd.dot(input_data[i], Wxh) + nd.dot(h, Whh) + bh
                h = nd.tanh(h_linear)
                yhat_linear = nd.dot(h, Why) + by
                yhat = softmax(yhat_linear) 
                outputs.append(nd.expand_dims(yhat[0],axis=1))
            return (outputs, h)
        
def list_to_nd_array(list_of_nd_arrays):
    return nd.concat(*list_of_nd_arrays)

# context usage
ctx = mx.cpu()
data, labels = load_time_machine()

num_hidden = 88
steps = 64
learning_rate = 0.01
vocab_size = 88

decoder_params = rnn_helper(num_hidden, vocab_size)
encoder_params = rnn_helper(num_hidden, vocab_size)
params = decoder_params + encoder_params

for epoch in range(100):
    for i in range(data.shape[0]):
        with autograd.record():
            output_encoder,hidden_encoder=encoder(steps, data[i], num_hidden, int(data.shape[3]), nd.zeros(num_hidden),encoder_params)
            out_enc = list_to_nd_array(output_encoder)
            output_decoder, hidden_state = decoder(steps,out_enc,nd.zeros(num_hidden),num_hidden,int(data.shape[3]),decoder_params)
            loss = average_ce_loss(output_decoder, nd.reshape(labels[i],(64,88))) 
        loss.backward()
        print decoder_params[0].grad
        SGD(params, learning_rate)