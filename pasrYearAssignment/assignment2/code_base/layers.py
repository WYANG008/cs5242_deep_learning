from builtins import range
import numpy as np
from timeit import default_timer as timer


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_rs = np.reshape(x, (N, -1))
    out = x_rs.dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    N = x.shape[0]
    x_rs = np.reshape(x, (N, -1))
    db = dout.sum(axis=0)
    dw = x_rs.T.dot(dout)
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = (x >= 0) * dout
    return dx

def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    if dropout_param is None or dropout_param.get('p') is None or dropout_param.get('p') == 0:
        return x, (dropout_param, None)
    
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = np.random.rand(*x.shape) > p
        out = x*mask/(1-p)
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    if dropout_param is None or dropout_param.get('p') is None or dropout_param.get('p') == 0:
        return dout
    p, mode = dropout_param['p'], dropout_param['mode']

    dx = None
    if mode == 'train':
        dx = dout*mask/(1-p)
    elif mode == 'test':
        dx = dout
    return dx


def padding(matrix, same_padding):
    """
    matrix is of size (c, h, w). The padded matrix will be of shapre (c, h+p, w+p)
    """
    return np.lib.pad(matrix, ((0,0),(0,0),(same_padding/2,same_padding/2),(same_padding/2,same_padding/2)), "constant",constant_values=((0,0),(0, 0),(0,0),(0,0)))

def get_vectorized_x(x, kh, kw, p, s):
    KH, KW = kh, kw
    N, C, XH, XW = x.shape
    OH = 1 + (XH + p - KH)/s
    OW = 1 + (XW + p - KW)/s
    x_reshaped = np.zeros((C*KH*KW, OH*OW*N))
    padded_x = padding(x, p)
    col = 0
    for n in range(N):
        i = j = 0
        while (i+KH <= XH + p):
            while (j + KW <= XW + p):
                x_reshaped[:,col] = padded_x[n,:,i:i+KH,j:j+KW].reshape(C*KH*KW)
                j += s
                col += 1
            j = 0
            i += s
    return x_reshaped

def conv_forward(x, w, b, conv_param):
    """
    Forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input in each x-y direction.
         We will use the same definition 
         in lecture notes 3b, slide 13 (ie. same padding on both sides).

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + pad - HH) / stride
      W' = 1 + (W + pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    s = conv_param['stride']
    p = conv_param['pad']
#   x: Input data of shape (N, C, H, W)
#   w: Filter weights of shape (F, C, HH, WW)
#   b: Biases, of shape (F,)
    F, C, KH, KW = w.shape
    N, C, XH, XW = x.shape
    OH = 1 + (XH + p - KH)/s
    OW = 1 + (XW + p - KW)/s
    filter_matrix = w.reshape((F, C*KH*KW))
    b_matrix = np.repeat(b.reshape((F,1)), OH*OW*N, axis=1)
    vectorized_x = get_vectorized_x(x, KH, KW, p, s)
    out = (np.dot(filter_matrix, vectorized_x)+b_matrix).reshape((F,N, OH,OW)).transpose(1,0,2,3)
    cache = (x, w, b, conv_param)
    return out, cache


def get_indexes(x_shape, KH, KW, p, s):
    N, C, H, W = x_shape
    OH = int((H + p - KH) / s + 1)
    OW = int((W + p - KW) / s + 1)

    i0 = np.tile(np.repeat(np.arange(KH), KW), C).reshape(-1, 1)
    i1 = s * np.repeat(np.arange(OH), OW).reshape(1, -1)
    i = i0 + i1

    j0 = np.tile(np.arange(KW), KH * C).reshape(-1, 1)
    j1 = s * np.tile(np.arange(OW), OH).reshape(1, -1)
    j = j0 + j1

    k = np.repeat(np.arange(C), KH * KW).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))

def get_conv_matrix(x, KH, FW, p, s):
    x_padded = padding(x,p)
    k, i, j = get_indexes(x.shape, KH, FW, p, s)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(KH * FW * C, -1)
    return cols

def get_x(cols, x_shape, KH, KW, p, s):
    # (C*KH*KW, OH*OW*N) => (N,C,XH,XW)
    N, C, H, W = x_shape
    x_padded = np.zeros((N, C, H + p, W + p), dtype=cols.dtype)
    k, i, j = get_indexes(x_shape, KH, KW, p, s)
    cols_reshaped = cols.reshape(C * KH * KW, -1, N).transpose(2, 0, 1) # N, C*KH*KW, OH*OW
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if p == 0:
        return x_padded
    return x_padded[:, :, p/2:-p/2, p/2:-p/2]

def conv_backward(dout, cache):
    """
    Backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    (x, w, b, conv_param) = cache
    s = conv_param['stride']
    p = conv_param['pad']
    
    dx = np.zeros(x.shape)   
    N, F, OH, OW = dout.shape
    F, C, KH, KW = w.shape
    N, C, XH, XW = x.shape
    
    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(F)
    
#   vectorized_x = get_vectorized_x(x, KH, KW, p, s)
    vectorized_x = get_conv_matrix(x, KH, KW, p, s)
    # Transpose from (N,F,OH,OW) into (F,OH,OW,N), then reshape into (F, OH*OW*N)
#     dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F,-1)
    dout_reshaped = dout.transpose(1,2,3,0).reshape(F,-1)
    # (F, OH*OW*N) x (OH*OW*N, C*KH*KW) = (F, C*KH*KW)
    dw = np.dot(dout_reshaped, vectorized_x.T)
    # Reshape back to (F, C, KH, KW)
    dw = dw.reshape(w.shape)
    
    # Reshape from F,C,KH,KW into (F,C*KH*KW)
    w_reshaped = w.reshape(F, -1)
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F,-1)
    # (C*KH*KW,F) x (F,OH*OW*N) = (C*KH*KW, OH*OW*N)
    dx_reshaped = np.dot(w_reshaped.T, dout_reshaped)
    # Stretched out image to the real image: (C*KH*KW, OH*OW*N) => (N,C,XH,XW)
    dx = get_x(dx_reshaped, x.shape, KH, KW, p, s)

    return dx, dw, db


def find_max(x):
    k = x.argmax()
    ncol = x.shape[1]
    position = (k/ncol, k%ncol)
    max_value = x[position]
    return position, max_value

def max_pool_forward(x, pool_param):
    """
    Forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param, loc)
    """
    out = None
    s = pool_param['stride']
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    N, C, H, W = x.shape
    OH = 1 + (H - PH)/s
    OW = 1 + (W - PW)/s
    vectorized_x = get_vectorized_x(x, PH, PW, 0, s)
    out = vectorized_x.reshape((C, PH*PW, OH*OW*N)).max(axis=1).reshape((C, N, OH, OW)).transpose(1,0,2,3)
    cache = (x, pool_param)
    return out, cache

def max_pool_backward(dout, cache):
    """
    Backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param, loc) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    (x, pool_param) = cache
    s = pool_param['stride']
    PH = pool_param['pool_height']
    PW = pool_param['pool_width']
    N, C, H, W = x.shape
    N, C, OH, OW = dout.shape
    dx = np.zeros(x.shape)
    vectorized_x = get_vectorized_x(x, PH, PW, 0, s)
    loc = vectorized_x.reshape((C, PH*PW, OH*OW*N)).argmax(axis=1).reshape((C, N, OH, OW)).transpose(1,0,2,3)
    row_loc = loc // PW
    col_loc = loc % PW
    cols = np.repeat(np.arange(OW), OH*N*C).reshape(OW, -1).T.reshape(N,C, OH, OW)
    rows = np.repeat(np.arange(OH), OW*N*C).reshape(OH, -1).reshape(OH, N, C, OW).transpose(1,2,0,3)
    rows = (rows - 1) * s + PH + row_loc
    cols = (cols - 1) * s + PW + col_loc
    for n in range(N):
        for c in range(C):
            for i in range(OH):
                for j in range(OW):
                    row = rows[n,c,i, j]
                    col = cols[n,c,i, j]
                    dx[n,c,row,col] = dout[n,c,i,j]
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
