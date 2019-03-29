pass
from code_base.layers import *
from timeit import default_timer as timer

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward(da, conv_cache)
    return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    # start = timer()
    a, conv_cache = conv_forward(x, w, b, conv_param)
    # end = timer()
    # print('conv forward: ', end - start, 'seconds')
    s, relu_cache = relu_forward(a)
    # start = timer()
    out, pool_cache = max_pool_forward(s, pool_param)
    # end = timer()
    # print('max pooling forward: ', end - start, 'seconds')
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    # start = timer()
    ds = max_pool_backward(dout, pool_cache)
    # end = timer()
    # print('max pooling backward: ', end - start, 'seconds')
    da = relu_backward(ds, relu_cache)
    # start = timer()
    dx, dw, db = conv_backward(da, conv_cache)
    # end = timer()
    # print('conv backward: ', end - start, 'seconds')
    return dx, dw, db
