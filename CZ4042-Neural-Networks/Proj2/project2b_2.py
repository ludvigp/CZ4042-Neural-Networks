from load import mnist
import numpy as np

import pylab

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# 1 encoder, decoder and a softmax layer

def init_weights(n_visible, n_hidden):
    initial_W = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)),
        dtype=theano.config.floatX)
    return theano.shared(value=initial_W, name='W', borrow=True)

def init_bias(n):
    return theano.shared(value=np.zeros(n,dtype=theano.config.floatX),borrow=True)

trX, teX, trY, teY = mnist()

trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]

x = T.fmatrix('x')  
d = T.fmatrix('d')


rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

corruption_level=0.1
training_epochs = 5
learning_rate = 0.1
batch_size = 128


W1 = init_weights(28*28, 900)
b1 = init_bias(900)
b1_prime = init_bias(28*28)
W1_prime = W1.transpose()

W2 = init_weights(900, 625)
b2 = init_bias(625)
b2_prime = init_bias(900)
W2_prime = W2.transpose()

W3 = init_weights(625, 400)
b3 = init_bias(400)
b3_prime = init_bias(625)
W3_prime = W3.transpose()

W4 = init_weights(400, 10)
b4 = init_bias(10)



tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*x
y1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
z1 = T.nnet.sigmoid(T.dot(y1, W1_prime) + b1_prime)

y2 = T.nnet.sigmoid(T.dot(y1, W2) + b2)
z2 = T.nnet.sigmoid(T.dot(y2, W2_prime) + b2_prime)

y3 = T.nnet.sigmoid(T.dot(y2, W3) + b3)
z3 = T.nnet.sigmoid(T.dot(y3, W3_prime) + b3_prime)

cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1))
cost2 = - T.mean(T.sum(y1 * T.log(z2) + (1 - y1) * T.log(1 - z2), axis=1))
cost3 = - T.mean(T.sum(y2 * T.log(z3) + (1 - y2) * T.log(1 - z3), axis=1))

params1 = [W1, b1, b1_prime]
grads1 = T.grad(cost1, params1)

updates1 = [(param1, param1 - learning_rate * grad1)
           for param1, grad1 in zip(params1, grads1)]
train_da1 = theano.function(inputs=[x], outputs = cost1, updates = updates1, allow_input_downcast = True)


params2 = [W2, b2, b2_prime]
grads2 = T.grad(cost2, params2)

updates2 = [(param2, param2 - learning_rate * grad2)
           for param2, grad2 in zip(params2, grads2)]
train_da2 = theano.function(inputs=[x], outputs = cost2, updates = updates2, allow_input_downcast = True)

params3 = [W3, b3, b3_prime]
grads3 = T.grad(cost3, params3)

updates3 = [(param3, param3 - learning_rate * grad3)
           for param3, grad3 in zip(params3, grads3)]
train_da3 = theano.function(inputs=[x], outputs = cost3, updates = updates3, allow_input_downcast = True)


##Softmax
p_y4 = T.nnet.softmax(T.dot(y3, W4)+b4)
y4 = T.argmax(p_y4, axis=1)
cost4 = T.mean(T.nnet.categorical_crossentropy(p_y4, d))

params4 = [W1, b1, W2, b2, W3, b3, W4, b4]
grads4 = T.grad(cost4, params4)
updates4 = [(param4, param4 - learning_rate * grad4)
           for param4, grad4 in zip(params4, grads4)]
train_ffn = theano.function(inputs=[x, d], outputs = cost4, updates = updates4, allow_input_downcast = True)
test_ffn = theano.function(inputs=[x], outputs = y4, allow_input_downcast=True)


print('training dae1 ...')
d = []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_da1(trX[start:end]))
    d.append(np.mean(c, dtype='float64'))
    print(d[epoch])


print("\ntraining dae2")
d = []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_da2(trX[start:end]))
    d.append(np.mean(c, dtype='float64'))
    print(d[epoch])


print("\ntraining dae3")
d = []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_da3(trX[start:end]))
    d.append(np.mean(c, dtype='float64'))
    print(d[epoch])


print('\ntraining ffn ...')
d, a = [], []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_ffn(trX[start:end], trY[start:end]))
    d.append(np.mean(c, dtype='float64'))
    a.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX)))
    print(a[epoch])

pylab.figure("Cross entropy")
pylab.plot(range(training_epochs), d)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.savefig('figure_2b_3.png')

pylab.figure("Test accuracy")
pylab.plot(range(training_epochs), a)
pylab.xlabel('iterations')
pylab.ylabel('test accuracy')
pylab.savefig('figure_2b_4.png')


pylab.show()