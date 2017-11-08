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

#Momentum
#Decay parameter ??
def sgd_momentum(cost, params, lr=0.1, decay=0.0001, momentum=0.1):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v = theano.shared(p.get_value())
        v_new = momentum*v - (g + decay*p) * lr
        updates.append([p, p + v_new])
        updates.append([v, v_new])
        return updates

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
beta = 0.5
rho = 0.05


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


tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*x
y1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
z1 = T.nnet.sigmoid(T.dot(y1, W1_prime) + b1_prime)

y2 = T.nnet.sigmoid(T.dot(y1, W2) + b2)
z2 = T.nnet.sigmoid(T.dot(y2, W2_prime) + b2_prime)

y3 = T.nnet.sigmoid(T.dot(y2, W3) + b3)
z3 = T.nnet.sigmoid(T.dot(y3, W3_prime) + b3_prime)


cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1)) \
				+ beta*T.shape(z1)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho)) \
				- beta*rho*T.sum(T.log(T.mean(z1, axis=0)+1e-6)) \
				- beta*(1-rho)*T.sum(T.log(1-T.mean(z1, axis=0)+1e-6))

cost2 = - T.mean(T.sum(y1 * T.log(z2) + (1 - y1) * T.log(1 - z2), axis=1)) \
				+ beta*T.shape(z2)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho)) \
				- beta*rho*T.sum(T.log(T.mean(z2, axis=0)+1e-6)) \
				- beta*(1-rho)*T.sum(T.log(1-T.mean(z2, axis=0)+1e-6))

cost3 = - T.mean(T.sum(y2 * T.log(z3) + (1 - y2) * T.log(1 - z3), axis=1)) \
				+ beta*T.shape(z1)[1]*(rho*T.log(rho) + (1-rho)*T.log(1-rho)) \
				- beta*rho*T.sum(T.log(T.mean(z3, axis=0)+1e-6)) \
				- beta*(1-rho)*T.sum(T.log(1-T.mean(z3, axis=0)+1e-6))


params1 = [W1, b1, b1_prime]
updates1 = sgd_momentum(cost1, params1)
train_da1 = theano.function(inputs=[x], outputs = cost1, updates = updates1, allow_input_downcast = True)

params2 = [W2, b2, b2_prime]
updates2 = sgd_momentum(cost2, params2)
train_da2 = theano.function(inputs=[x], outputs = cost2, updates = updates2, allow_input_downcast = True)

params3 = [W3, b3, b3_prime]
updates3 = sgd_momentum(cost3, params3)
train_da3 = theano.function(inputs=[x], outputs = cost3, updates = updates3, allow_input_downcast = True)


print('training dae1 ...')
d = []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_da1(trX[start:end]))
    d.append(np.mean(c, dtype='float64'))
    print(d[epoch])

pylab.figure("Cross entropy hidden layers")
pylab.plot(range(training_epochs), d, color = "blue", label = "Training layer 1")
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')

print("\ntraining dae2")
d = []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_da2(trX[start:end]))
    d.append(np.mean(c, dtype='float64'))
    print(d[epoch])


pylab.plot(range(training_epochs), d, color = "red", label = "Training layer 2")


print("\ntraining dae3")
d = []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_da3(trX[start:end]))
    d.append(np.mean(c, dtype='float64'))
    print(d[epoch])

pylab.plot(range(training_epochs), d, color = "green", label = "Training layer 3")
pylab.plt.legend(loc = "best")


w1 = W1.get_value()
pylab.figure('Hidden Layer 1')
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w1[:,i].reshape(28,28))
pylab.savefig('figure_2b_2.png')


w2 = W2.get_value()
pylab.figure('Hidden Layer 2')
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w2[:,i].reshape(30,30))
pylab.savefig('figure_2b_5.png')

w3 = W3.get_value()
pylab.figure('Hidden Layer 3')
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w3[:,i].reshape(25,25))
pylab.savefig('figure_2b_6.png')


pylab.show()