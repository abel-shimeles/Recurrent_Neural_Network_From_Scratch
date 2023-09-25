import numpy as np

def softmax(z):
    e_x = np.exp(z - np.max(z))
    output = e_x / e_x.sum(axis=0)

    return output

def sigmoid(z):
    output = 1 / (1 + np.exp(-z))

    return output

def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        v['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)
        s['dW' + str(l + 1)] = np.zeros(parameters['W' + str(l + 1)].shape)
        s['db' + str(l + 1)] = np.zeros(parameters['b' + str(l + 1)].shape)

    return v, s

def adam_update_parameters(parameters, grads, v, s, t, learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    
    for l in range(L):
        v['dW' + str(l + 1)] = beta_1 * v['dW' + str(l + 1)] + (1 - beta_1) * grads['dW' + str(l + 1)]
        v['db' + str(l + 1)] = beta_1 * v['db' + str(l + 1)] + (1 - beta_1) * grads['db' + str(l + 1)]

        v_corrected['dW' + str(l + 1)] = v['dW' + str(l + 1)] / (1 - beta_1 ** t)
        v_corrected['db' + str(l + 1)] = v['db' + str(l + 1)] / ( 1- beta_1 ** t)

        s['dW' + str(l + 1)] = beta_2 * s['dW' + str(l + 1)] + (1 - beta_2) * (grads['db' + str(l + 1)] ** 2)
        s['db' + str(l + 1)] = beta_2 * s['db' + str(l + 1)] + (1 - beta_2) * (grads['db' + str(l + 1)] ** 2)

        s_corrected['dW' + str(l + 1)] = s['dW' + str(l + 1)] / (1 - beta_2 ** t)
        s_corrected['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - beta_2 ** t)

        parameters['W' + str(l + 1)] = parameters['W' + str(l + 1)] - (learning_rate * (v['dW' + str(l + 1)] / ((np.sqrt(s_corrected['dW' + str(l + 1)]) + epsilon))))
        parameters['b' + str(l + 1)] = parameters['b' + str(l + 1)] - (learning_rate * (v['db' + str(l + 1)] / ((np.sqrt(s_corrected['db' + str(l + 1)]) + epsilon))))

        return parameters, v, s
    

# Forward Propagation for a basic RNN

def rnn_cell_forward(x_t, a_prev, parameters):
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']
    
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x_t) + ba)
    y_t_pred = softmax(np.dot(Wya, a_next) + by)

    cache = (a_next, a_prev, x_t, parameters)

    return a_next, y_t_pred, cache

def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, x_t, parameters) = cache
    
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    dtanh = (da_next * (1 - (np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, x_t) + ba) ** 2)))
    dxt = np.dot(Wax.T, dtanh)
    da_prev = np.dot(Waa.T, dtanh)
    dWax = np.dot(dtanh, x_t.T)
    dWaa = np.dot(dtanh, a_prev.T)
    dba = np.sum(dtanh, axis=1, keepdims=True)

    gradients = {
        'dxt': dxt,
        'da_prev': da_prev,
        'dWax': dWax,
        'dWaa': dWaa,
        'dba': dba,
    }

    return gradients

def rnn_backward(da, caches):
    (caches, x) = caches
    (a_1, a_0, x_1, parameters) = caches[0]

    n_a, m, T_x = da.shape
    n_x, m = x_1.shape

    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da_0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))

    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da_prevt, caches)
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients['dxt'], gradients['da_prev'], gradients['dWax'], gradients['dWaa'], gradients['dba']
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    da_0 = da_prevt

    gradients = {
        'dxt': dxt,
        'da_0': da_0,
        'dWax': dWax,
        'dWaa': dWaa,
        'dba': dba,
    }

    return gradients



def rnn_forward(x, a_0, parameters):
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape

    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    a_next = a_0

    for t in range(T_x):
        a_next, y_t_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = y_t_pred
        caches.append(cache)
    
    caches = (caches, x)

    return a, y_pred, caches

# Long Short-Term Memory(LSTM) Network

def lstm_cell_forward(x_t, a_prev, c_prev, parameters):
    Wf = parameters['Wf']
    bf = parameters['bf']
    Wi = parameters['Wi']
    bi = parameters['bi']
    Wc = parameters['Wc']
    bc = parameters['bc']
    Wo = parameters['Wo']
    bo = parameters['bo']
    Wy = parameters['Wy']
    by = parameters['by']

    n_x, m = x_t.shape
    n_y, n_a = Wy.shape

    concat = np.concatenate([a_prev, x_t], axis=0)

    cc_t = np.tanh(np.dot(Wc, concat) + bc)
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    c_next = (it * cc_t) + (ft * c_prev)
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = ot * np.tanh(c_next)

    y_t_pred = softmax(np.dot(Wy, a_next) + by)

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cc_t, ot, x_t, parameters)

    return a_next, c_next, y_t_pred, cache

def lstm_forward(x, a_0, parameters):
    caches = []

    Wy = parameters['Wy']
    n_x, m ,T_x = x.shape
    n_y, n_a = Wy.shape

    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    a_next = a_0
    c_next = np.zeros((n_a, m))

    for t in range(T_x):
        x_t = x[:, :, t]
        a_next, c_next, y_t_pred, cache = lstm_cell_forward(x_t, a_next, c_next, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = y_t_pred

        caches.append(cache)

    caches = (caches, x)

    return a, y, c, caches

def lstm_cell_backward(da_next, dc_next, cache):
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache
    
    n_x, m = xt.shape
    n_a, m = a_next.shape
    
    dot = da_next*np.tanh(c_next)*ot*(1-ot)
    dcct = (dc_next*it+ot*(1-np.square(np.tanh(c_next)))*it*da_next)*(1-np.square(cct))
    dit = (dc_next*cct+ot*(1-np.square(np.tanh(c_next)))*cct*da_next)*it*(1-it)
    dft = (dc_next*c_prev+ot*(1-np.square(np.tanh(c_next)))*c_prev*da_next)*ft*(1-ft)

    dWf = np.dot(dft,np.concatenate((a_prev, xt), axis=0).T)
    dWi = np.dot(dit,np.concatenate((a_prev, xt), axis=0).T)
    dWc = np.dot(dcct,np.concatenate((a_prev, xt), axis=0).T)
    dWo = np.dot(dot,np.concatenate((a_prev, xt), axis=0).T)
    dbf = np.sum(dft,axis=1,keepdims=True)
    dbi = np.sum(dit,axis=1,keepdims=True)
    dbc = np.sum(dcct,axis=1,keepdims=True)
    dbo = np.sum(dot,axis=1,keepdims=True)

    da_prev = np.dot(parameters['Wf'][:,:n_a].T,dft)+np.dot(parameters['Wi'][:,:n_a].T,dit)+np.dot(parameters['Wc'][:,:n_a].T,dcct)+np.dot(parameters['Wo'][:,:n_a].T,dot)
    dc_prev = dc_next*ft+ot*(1-np.square(np.tanh(c_next)))*ft*da_next
    dxt = np.dot(parameters['Wf'][:,n_a:].T,dft)+np.dot(parameters['Wi'][:,n_a:].T,dit)+np.dot(parameters['Wc'][:,n_a:].T,dcct)+np.dot(parameters['Wo'][:,n_a:].T,dot)
    
    gradients = {
        "dxt": dxt, 
        "da_prev": da_prev, 
        "dc_prev": dc_prev, 
        "dWf": dWf,
        "dbf": dbf, 
        "dWi": dWi,
        "dbi": dbi,
        "dWc": dWc,
        "dbc": dbc, 
        "dWo": dWo,
        "dbo": dbo
    }

    return gradients

def lstm_backward(da, caches):
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]
    
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = np.zeros((n_a, m))
    dc_prevt = np.zeros((n_a, m))
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))

    for t in reversed(range(T_x)):
        gradients = lstm_cell_backward(da[:,:,t] + da_prevt, dc_prevt, caches[t])
        da_prevt = gradients['da_prev']
        dc_prevt = gradients['dc_prev']
        dx[:,:,t] = gradients['dxt']
        dWf += gradients['dWf']
        dWi += gradients['dWi']
        dWc += gradients['dWc']
        dWo += gradients['dWo']
        dbf += gradients['dbf']
        dbi += gradients['dbi']
        dbc += gradients['dbc']
        dbo += gradients['dbo']

    da0 = da_prevt

    gradients = {
        "dx": dx, 
        "da0": da0,
        "dWf": dWf,
        "dbf": dbf, 
        "dWi": dWi,
        "dbi": dbi,
        "dWc": dWc,
        "dbc": dbc, 
        "dWo": dWo,
        "dbo": dbo
    }
    
    return gradients


