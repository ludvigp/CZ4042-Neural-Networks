from load import mnist
import numpy as np
import pylab

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

# 2 convolution layer, 2 max pooling layer, one fully connected layer and a softmax layer

np.random.seed(10)
batch_size = 128
noIters = 25


def init_weights_bias4(filter_shape, d_type):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])
    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values = np.asarray(
        np.random.uniform(low=-bound, high=bound, size=filter_shape),
        dtype=d_type)
    b_values = np.zeros((filter_shape[0],), dtype=d_type)
    return theano.shared(w_values, borrow=True), theano.shared(b_values, borrow=True)


def init_weights_bias2(filter_shape, d_type):
    fan_in = filter_shape[1]
    fan_out = filter_shape[0]

    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values = np.asarray(
        np.random.uniform(low=-bound, high=bound, size=filter_shape),
        dtype=d_type)
    b_values = np.zeros((filter_shape[1],), dtype=d_type)
    return theano.shared(w_values, borrow=True), theano.shared(b_values, borrow=True)


def model(X, w1, b1, w2, b2, w3, b3, w4, b4):
    y1 = T.nnet.relu(conv2d(X, w1) + b1.dimshuffle('x', 0, 'x', 'x'))
    pool_dim = (2, 2)
    o1 = pool.pool_2d(y1, pool_dim)

    # second convolutional and pooling layers
    y2 = T.nnet.relu(conv2d(o1, w2) + b2.dimshuffle('x', 0, 'x', 'x'))
    o2 = pool.pool_2d(y2, pool_dim)

    # flatten the second pooling output
    o3 = T.flatten(o2, outdim=2)

    # make the network fully connected, having a layer of 100 neurons
    y3 = T.nnet.relu(T.dot(o3, w3) + b3)

    # forward the output of fully connected layer to the 10 classification neurons in py_x,
    # activation function is now the softmax for probabilities in the outputs
    py_x = T.nnet.softmax(T.dot(y3, w4) + b4)
    return y1, o1, y2, o2, y3, o3, py_x


def sgd_momentum(cost, params, lr=0.05, decay=0.0001, momentum=0.1):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v = theano.shared(p.get_value()*0.)
        v_new = momentum*v - (g + decay*p) * lr
        updates.append([p, p + v])
        updates.append([v, v_new])
    return updates


def shuffle_data(samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels


trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]

X = T.tensor4('X')
Y = T.matrix('Y')

num_filters1 = 15
num_filters2 = 20
fully_connected_layer = 100

# Init weights and biases for convolution layer, 25 filters and filters of window size 9x9
# Input weights, first conv. 15 matrixes of 9x9 weights, 1 vector of 25 biases
w1, b1 = init_weights_bias4((num_filters1, 1, 9, 9), X.dtype)

# weigths for second convolutional layer. 20 matrixes by 5x5. These are to be applied to 15 pooled pictures from layer1.
w2, b2 = init_weights_bias4((num_filters2, num_filters1, 5, 5), X.dtype)

# weights to fully connected layer, 3x3 is is the output from the pooling layer, num_filters2 times
# 180 intputs to fully connected layer, 100 outputs
w3, b3 = init_weights_bias2((num_filters2 * 3 * 3, fully_connected_layer), X.dtype)

# 100 inputs to softmax layer, 10 outputs of probabilities
w4, b4 = init_weights_bias2((fully_connected_layer, 10), X.dtype)

y1, o1, y2, o2, y3, o3, py_x = model(X, w1, b1, w2, b2, w3, b3, w4, b4)

y_x = T.argmax(py_x, axis=1)
cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w1, b1, w2, b2, w3, b3, w4, b4]
updates = sgd_momentum(cost, params, lr=0.05)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)

predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

test = theano.function(inputs=[X], outputs=[y1, o1], allow_input_downcast=True)

test2 = theano.function(inputs=[X], outputs=[y2, o2], allow_input_downcast=True)

# accuracy and cost
a = []
c = []
for i in range(noIters):
    trX, trY = shuffle_data(trX, trY)
    teX, teY = shuffle_data(teX, teY)
    cost = 0.0
    n = len(trX)
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost += train(trX[start:end], trY[start:end])
    c = np.append(c, cost / (n // batch_size))
    a.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
    print("Iteration " + str(i) + ": " + str(a[i]))

# plot test accuracy vs epochs
pylab.figure()
pylab.plot(range(noIters), a)
pylab.xlabel('epochs')
pylab.ylabel('test accuracy')
pylab.savefig('figure_2a2_accuracy.png')

pylab.figure()
pylab.plot(range(noIters), c)
pylab.xlabel('epochs')
pylab.ylabel('cost')
pylab.savefig('figure_2a2_cost.png')

w = w1.get_value()
pylab.figure('filters learned')
pylab.gray()
for i in range(num_filters1):
    pylab.subplot(5, 3, i + 1); pylab.axis('off'); pylab.imshow(w[i, :, :, :].reshape(9, 9))
pylab.savefig('figure_2a2_filtersLearned.png')

ind = np.random.randint(low=0, high=2000)
convolved, pooled = test(teX[ind:ind + 1, :])
ind = np.random.randint(low=0, high=2000)
convolved2, pooled2 = test2(teX[ind:ind + 1, :])

pylab.figure('input image')
pylab.gray()
pylab.axis('off');
pylab.imshow(teX[ind, :].reshape(28, 28))
pylab.savefig('figure_2a2_inputImage.png')

pylab.figure('convolved feature maps_1')
pylab.gray()
for i in range(num_filters1):
    pylab.subplot(5, 3, i + 1); pylab.axis('off'); pylab.imshow(convolved[0, i, :].reshape(20, 20))
pylab.savefig('figure_2a2_conv1.png')

pylab.figure('pooled feature maps_1')
pylab.gray()
for i in range(num_filters1):
    pylab.subplot(5, 3, i + 1); pylab.axis('off'); pylab.imshow(pooled[0, i, :].reshape(10, 10))
pylab.savefig('figure_2a2_pool1.png')

pylab.figure('convolved_2')
pylab.gray()
for i in range(num_filters2):
    pylab.subplot(5, 4, i + 1); pylab.axis('off'); pylab.imshow(convolved2[0, i, :].reshape(6, 6))
pylab.savefig('figure_2a2_conv2.png')

pylab.figure('pool_2')
pylab.gray()
for i in range(num_filters2):
    pylab.subplot(5, 4, i + 1); pylab.axis('off'); pylab.imshow(pooled2[0, i, :].reshape(3, 3))
pylab.savefig('figure_2a2_pool2.png')

pylab.show()

# Hvorfor får vi så av accuracy? Sigmoid eller ReLu
# Plotte weights i tillegg til conv og pools? why
# Implemented second conv layer, lower starting accuracy
# Riktig cost-funksjon??
