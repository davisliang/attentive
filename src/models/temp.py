
def attention_helper(num_attention, num_hidden_encoder, num_hidden_decoder):
    W = nd.random_normal(shape=(num_attention,num_hidden_decoder), ctx=ctx) * .01
    V = nd.random_normal(shape=(num_attention,num_hidden_encoder), ctx=ctx) * .01
    w =  nd.random_normal(shape=(num_attention,1), ctx=ctx) * .01
    b = nd.random_normal(shape=num_attention, ctx=ctx) * .01
    params = [W,V,w,b]
    return params

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

def attention(decoder_hidden, encoder_output, att_params):
    W, V, w, b = att_params
    decoder_temp = nd.dot(W,decoder_hidden)
    encoder_temp = nd.dot(V,encoder_output)
    net_temp = decoder_temp+encoder_temp+b
    return nd.dot(w,nd.tanh(net_temp))
    #return nd.dot(softmax(nd.dot(decoder_hidden_t, encoder_output)) , encoder_output.T)

def decoder(steps, encoder_outputs, state, num_hidden, vocab_size, params, att_params):
    Wxh, Whh, bh, Why, by = params
    outputs = []
    h = state
    # only look at steps long. (consider this 'dynamic')
    for i in range(steps):
        #h=nd.reshape(h,(1,h.size))
        yhat = softmax(nd.dot(nd.tanh(nd.dot(attention(h, encoder_outputs, att_params), Wxh) + nd.dot(h, Whh) + bh), Why) + by) 
        outputs.append(yhat[0])
    return (outputs, h)    
