import numpy as np
from skimage.util.shape import view_as_windows


#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
                                  (num_filters, in_ch_size, filter_width, filter_height))
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

        #######
        ## If necessary, you can define additional class variables here
        #######

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
        out_size = input_size - filter_width + 1  # set out_size

        y = view_as_windows(x, (1, in_ch_size, filter_width, filter_height))
        y = y.reshape((batch_size, out_size, out_size, -1))

        w = self.W.reshape((num_filters, -1, 1))
        result = y.dot(w)
        result = np.squeeze(result, axis=4)
        result = np.swapaxes(result, 3, 1)
        result = np.swapaxes(result, 2, 3)
        out = result
        out += self.b
        #print('######## forward x  #########\n',x)
        #print('######## forward W  #########\n', self.W)
        #print('######## forward b  #########\n', self.b)
        #print('######## forward out  #########\n', out)
        return out

    #######
    # Q2. Complete this method
    #######
    def backprop(self, x, dLdy):
        #print(dLdy.shape)   8.8.30.30
        (y_b, y_fn, y_w, y_h) = dLdy.shape

        ############ dLdb #############
        dLdb = np.zeros_like(self.b)
        for i in range(batch_size):
            for j in range(num_filters):
                dLdb[0][j] += np.sum(dLdy[i][j])
        ##############################

        ############ dLdW #############
        filt = dLdy.reshape(batch_size * num_filters, 1, y_w, y_h)
        window = view_as_windows(x, (1, 1, y_w, y_h))
        window = window.reshape((batch_size*in_ch_size, filter_width, filter_height, -1))
        filt = filt.reshape((batch_size * num_filters, -1, 1))
        result = window.dot(filt)
        result = np.squeeze(result, axis=4)
        result = np.swapaxes(result, 3, 1)
        result = np.swapaxes(result, 2, 3)

        result = result.reshape(batch_size * in_ch_size, batch_size * num_filters, filter_width, filter_height)
        batch_sum = np.zeros((in_ch_size, batch_size * num_filters, filter_width, filter_height))

        for i in range(batch_size):
            batch_sum += result[i * in_ch_size:i * in_ch_size + in_ch_size]

        batch2_sum = np.zeros((in_ch_size, num_filters, filter_width, filter_height))

        for i in range(in_ch_size):
            for j in range(batch_size):
                batch2_sum[i] += batch_sum[i][j * num_filters:j * num_filters + num_filters]

        final = batch2_sum / batch_size
        dLdW = np.swapaxes(final, 0, 1)
        #################################

        ############# dLdx ##############
        padding = 2
        dLdy = np.pad(dLdy, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant', constant_values=0)
        window2 = view_as_windows(dLdy, (1, 1, filter_width, filter_height))
        window2 = window2.reshape((batch_size , num_filters, input_size, input_size, -1))
        filt2 = np.flip(self.W,axis=3)
        filt2 = np.flip(filt2, axis= 2)
        filt2 = filt2.reshape((num_filters,in_ch_size, -1, 1))
        dLdx = np.zeros_like(x)
        for i in range(batch_size):
            for j in range(num_filters):
                for k in range(in_ch_size):
                    dot = window2[i][j].dot(filt2[j][k])
                    result2 = np.squeeze(dot, axis=2)
                    dLdx[i][k] += result2
        return dLdx, dLdW, dLdb

    #######
    ## If necessary, you can define additional class methods here
    #######


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######

    #######
    # Q3. Complete this method
    #######
    def forward(self, x):
        out_size = int((input_size - self.pool_size)/self.stride) +1
        x_to_2d = x.reshape(batch_size * in_ch_size * input_size, input_size)
        y = view_as_windows(x_to_2d, (self.pool_size, self.pool_size), step = self.stride)
        y = y.reshape(batch_size,in_ch_size,out_size,out_size,-1)
        out = np.max(y, axis=4)

        return out

    #######
    # Q4. Complete this method
    #######
    def backprop(self, x, dLdy):
        out_size = int((input_size - self.pool_size) / self.stride) + 1
        dLdx = np.zeros_like(x,dtype=np.float64)

        for i in range(batch_size):
            for j in range(in_ch_size):
                for k in range(out_size):
                    for l in range(out_size):
                        temp = x[i][j][k*self.pool_size:k*self.pool_size+self.pool_size,l*self.pool_size:l*self.pool_size+self.pool_size]
                        temp2 = np.where(temp == np.max(temp), 1, 0)
                        temp2 = temp2 * dLdy[i][j][k][l]
                        dLdx[i][j][k*self.pool_size:k*self.pool_size+self.pool_size,l*self.pool_size:l*self.pool_size+self.pool_size] = temp2
        return dLdx


    #######
    ## If necessary, you can define additional class methods here
    #######


# testing the implementation

# data sizes
batch_size = 8
input_size = 32
filter_width = 3
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    x = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) / exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')