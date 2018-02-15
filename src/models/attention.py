import mxnet as mx
import numpy as np
from mxnet import nd, gluon, autograd
from mxnet.gluon import nn, Block

def one_hots(numerical_list, vocab_size, data_ctx):
    result = nd.zeros((len(numerical_list), vocab_size), ctx=data_ctx)
    for i, idx in enumerate(numerical_list):
        result[i, idx] = 1.0
    return result

def textify(embedding):
    result = ""
    indices = nd.argmax(embedding, axis=0).asnumpy()
    for idx in indices:
        result += character_list[int(idx)]
    return result

def load_time_machine(seq_length=64, batch_size=1, data_ctx=mx.cpu()):
    # loading dataset
    path = "../../data/timemachine.txt"
    with open(path) as f:
        time_machine = f.read()
    time_machine = time_machine[:-38083] #hardcoded to remove crap
    character_dict, vocab_size = get_char_dict(time_machine)
    
    time_numerical = [character_dict[char] for char in time_machine]
    # -1 here so we have enough characters for labels later
    num_samples = (len(time_numerical) - 1) // seq_length
    dataset = one_hots(time_numerical[:seq_length*num_samples],vocab_size, data_ctx).reshape((num_samples, seq_length, vocab_size))
    num_batches = len(dataset) // batch_size
    train_data = dataset[:num_batches*batch_size].reshape((batch_size, num_batches, seq_length, vocab_size))
    
    # swap batch_size and seq_length axis to make later access easier
    train_data = nd.swapaxes(train_data, 0, 1)
    train_data = nd.swapaxes(train_data, 1, 2)
    print('Shape of data set: ', train_data.shape)
    
    labels = one_hots(time_numerical[1:seq_length*num_samples+1], vocab_size,data_ctx)
    train_label = labels.reshape((batch_size, num_batches, seq_length, vocab_size))
    train_label = nd.swapaxes(train_label, 0, 1)
    train_label = nd.swapaxes(train_label, 1, 2)
    print('Shape of label set: ', train_label.shape)
    
    return train_data, train_label

def load_english_to_french(seq_length=64,batch_size=1):
    #2600L, 64L, 1L, 88L
    path = "../../data/"

def get_char_dict(data):
    # get character dictionary
    character_list = list(set(data))
    vocab_size = len(character_list)
    # get the character dictionary
    character_dict = {}
    for e, char in enumerate(character_list):
        character_dict[char] = e
    return character_dict, vocab_size

def get_char_dict_builder(data, character_dict):
    # get character dictionary
    print "building dictionary"
    for line in data:
        character_list = list(set(line))
        # get the character dictionary
        for i in range(len(character_list)):
            if(character_list[i] not in character_dict):
                character_dict[character_list[i]] = len(character_dict)
    vocab_size = len(character_dict)
    return character_dict, vocab_size

def rnn_helper(num_hidden, vocab_size, model_ctx): 
    num_inputs = vocab_size
    num_outputs = vocab_size
    Wxh = nd.random_normal(1,0,shape=(num_inputs,num_hidden), ctx=model_ctx) 
    Whh = nd.random_normal(1,0,shape=(num_hidden,num_hidden), ctx=model_ctx) 
    bh = nd.random_normal(1,0,shape=num_hidden, ctx=model_ctx) 
    Why = nd.random_normal(1,0,shape=(num_hidden,num_outputs), ctx=model_ctx) 
    by = nd.random_normal(1,0,shape=num_outputs, ctx=model_ctx) 
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
    input_data = input_data.as_in_context(Wxh.context)
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
        return nd.dot(softmax(nd.dot(decoder_hidden_t, nd.reshape(encoder_output,(encoder_output.shape[1], encoder_output.shape[0])))), encoder_output)
    return nd.dot(softmax(nd.dot(decoder_hidden_t, encoder_output)) , nd.reshape(encoder_output,(encoder_output.shape[1], encoder_output.shape[0])))
 
def decoder(steps, encoder_outputs, state, num_hidden, vocab_size, params):
    Wxh, Whh, bh, Why, by = params
    outputs = []
    h = state
    # only look at steps long. (consider this 'dynamic')
    for i in range(steps):
        h=nd.reshape(h,(1,h.size))
        yhat = softmax(nd.dot(nd.tanh(nd.dot(attention(h, encoder_outputs), Wxh) + nd.dot(h, Whh) + bh), Why) + by) 
        outputs.append(yhat[0])
    return (outputs, h)    
    
def SGD(params, lr):    
    for param in params:
        param[:] = param - lr * param.grad
        

def cross_entropy(yhat, y, ctx):
    yhat = yhat.as_in_context(ctx)
    y = y.as_in_context(ctx)

    return - nd.sum(y * nd.log(yhat), axis=0, exclude=True)


def average_ce_loss(outputs, labels, ctx):
    assert(len(outputs) == len(labels))
    total_loss = 0.
    for (output, label) in zip(outputs,labels):
        total_loss = total_loss + cross_entropy(output, label, ctx)
    return total_loss / len(outputs)

        
def list_to_nd_array(list_of_nd_arrays):
    return nd.concat(*list_of_nd_arrays)

def list_to_nd_array_with_reshaping(list_of_nd_arrays):
    for i in range(len(list_of_nd_arrays)):
        list_of_nd_arrays[i]=list_of_nd_arrays[i].reshape((list_of_nd_arrays[i].shape[0],1))
    return nd.concat(*list_of_nd_arrays)


def translation_numerical(data,character_dict):
    print "turning characters into numerical representation"
    return_list=[]
    for line in data:
        return_list.append([character_dict[char] for char in line])
    return return_list

def numerical_to_nd(one_data,translation_dict, data_ctx):
    one_hot = one_hots(one_data, len(translation_dict), data_ctx)
    temp = one_hot.reshape((1,1,one_hot.shape[0],one_hot.shape[1]))
    temp = nd.swapaxes(temp,0,1)
    temp = nd.swapaxes(temp,1,2)
    return temp

def clean_data(train_data, test_data, threshold_min, threshold_max):
    print "cleaning data"
    train_data_list = []
    test_data_list = []
    for train_line, test_line in zip(train_data,test_data):
            train_line = train_line.lower()
            test_line = test_line.lower()  
            return_train_line = ""
            return_test_line = ""
            
            for i in range(len(train_line)):
                c = train_line[i]
                #if((ord(c)>=32 and ord(c)<=63) or (ord(c)>=96 and ord(c)<=127)):
                if((ord(c)==32) or (ord(c)>=97 and ord(c)<=122)):
                    return_train_line = return_train_line + c
                    
            for i in range(len(test_line)):
                c = test_line[i]
                if((ord(c)==32) or (ord(c)>=97 and ord(c)<=122)):
                    return_test_line = return_test_line + c
            
            if(len(return_train_line)>=threshold_min and len(return_train_line)<=threshold_max):
                train_data_list.append(return_train_line)
                test_data_list.append(return_test_line)
    return train_data_list,test_data_list

def pad_zeros(data_numerical):
    print "padding zeros"
    #first, find the maximum length of data.
    max_len = 0
    for line in data_numerical:
        if(len(line)>max_len):
            max_len = len(line)
            
    #iterate through each line and pad with zeros until length equals max_len
    for i in range(len(data_numerical)):
        data_numerical[i] = data_numerical[i] + [0]*(max_len - len(data_numerical[i]))
    
    return data_numerical

data_ctx = mx.cpu()
model_ctx = mx.gpu()

# open the datasets
with open("../../data/train.en","rb") as f:
    raw_train_data = f.read().splitlines()
with open("../../data/train.fr","rb") as f:
    raw_train_labels = f.read().splitlines()

#clean data
train_data, train_labels = clean_data(raw_train_data, raw_train_labels, 100,200)

# create dictionary and a character list 
translation_dict = {}
_, num_items = get_char_dict_builder(train_data,translation_dict)
_, num_items = get_char_dict_builder(train_labels, translation_dict)
character_list = list(translation_dict.keys())
print("vocabulary: ",character_list)
# from characters to numerical representations
english_numerical=translation_numerical(train_data,translation_dict)
french_numerical=translation_numerical(train_labels,translation_dict)

# pad zeros
#data = pad_zeros(english_numerical)
#labels = pad_zeros(french_numerical)
data = english_numerical
labels = french_numerical

num_hidden = len(translation_dict)
learning_rate = 0.01
vocab_size = len(translation_dict)

decoder_params = rnn_helper(num_hidden, vocab_size, model_ctx)
encoder_params = rnn_helper(num_hidden, vocab_size, model_ctx)
params = decoder_params + encoder_params

for epoch in range(100):
    for i in range(len(data)):
        with autograd.record():
            en = numerical_to_nd(data[i],translation_dict, data_ctx)
            fr = numerical_to_nd(labels[i],translation_dict, data_ctx)
            en = en.reshape((en.shape[1],en.shape[2],en.shape[3]))
            fr = fr.reshape((fr.shape[1],fr.shape[2],fr.shape[3]))
            
            output_encoder,hidden_encoder=encoder(en.shape[0], en, num_hidden, int(en.shape[2]), nd.zeros(num_hidden,ctx=model_ctx),encoder_params)
            out_enc = list_to_nd_array(output_encoder)
            output_decoder, hidden_state = decoder(fr.shape[0],out_enc,nd.zeros(num_hidden,ctx=model_ctx),num_hidden,int(fr.shape[2]),decoder_params)
            loss = average_ce_loss(output_decoder, nd.reshape(fr,(fr.shape[0],fr.shape[2])),ctx=model_ctx) 

        loss.backward()
        SGD(params, learning_rate)
        
        if(i%100==0):
            print("PRED: ", output_decoder[0])
            print("TARG: ", fr[0])
            print("grad: ", decoder_params[0])
            print("pred text: ",textify(list_to_nd_array_with_reshaping(output_decoder)))
            print("loss: ", loss)
