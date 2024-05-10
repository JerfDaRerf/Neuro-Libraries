import torch

#Computes the loss and gradient for softmax classification.
def softmax_loss(x, y):
    
    shifted_logits = x - x.max(dim=1, keepdim=True).values
    Z = shifted_logits.exp().sum(dim=1, keepdim=True)
    log_probs = shifted_logits - Z.log()
    probs = log_probs.exp()
    N = x.shape[0]
    loss = (-1.0 / N) * log_probs[torch.arange(N), y].sum()
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N
    return loss, dx

class Linear(object):

    # Computes the forward pass for an linear (fully-connected) layer.
    @staticmethod
    def forward(x, w, b):
        
        x_flat = torch.flatten(x, start_dim=1)
        product = torch.mm(x_flat, w)
        out = torch.add(product, b)
        cache = (x, w, b)
        return out, cache

    #Computes the backward pass for an linear layer.
    @staticmethod
    def backward(dout, cache):
        x, w, b = cache
        dx, dw, db = None, None, None
        # Convert to torch tensors if not already
        x, w, b = map(torch.tensor, (x, w, b))
    
        N = x.size(0)
    
        # Gradient with respect to x
        dx = dout.mm(w.t()).view(x.size())
    
        # Gradient with respect to w
        dw = x.view(N, -1).t().mm(dout)
    
        # Gradient with respect to b
        db = dout.sum(dim=0)
        return dx, dw, db

class ReLU(object):

    # Computes the forward pass for a layer of rectified
    #     linear units (ReLUs).
    #     Input:
    #     - x: Input; a tensor
    #     Returns a tuple of:
    #     - out: Output, a tensor of the same shape as x
    #     - cache: x
    @staticmethod
    def forward(x):
        
        out = None
        out = x.clone()
        out[out <= 0] = 0
      
        cache = x
        return out, cache

    @staticmethod
    #Computes the backward pass for a layer of rectified linear units (ReLUs).
    def backward(dout, cache):
        

        dx, x = None, cache


        x_c = x.clone()
        x_c[x_c > 0] = 1
        x_c[x_c < 0] = 0
        
        dx = torch.mul(dout, x_c)

        return dx


class Linear_ReLU(object):

    # Convenience layer that performs an linear transform
    # followed by a ReLU.
    @staticmethod
    def forward(x, w, b):
     
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    # Backward pass for the linear-relu convenience layer
    @staticmethod
    def backward(dout, cache):

        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db

# A two-layer fully-connected neural network with ReLU nonlinearity and
# softmax loss that uses a modular layer design.
class TwoLayerNet(object):
  

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cpu'):
      
        self.params = {}
        self.reg = reg
        self.device = device
        
        W1 = torch.normal(mean=0.0, std=weight_scale, size=(input_dim, hidden_dim), dtype=dtype, device=self.device)
        b1 = torch.zeros(hidden_dim, dtype=dtype, device=self.device)
        W2 = torch.normal(mean=0.0, std=weight_scale, size=(hidden_dim, num_classes), dtype=dtype, device=self.device)       
        b2 = torch.zeros(num_classes, dtype=dtype, device=self.device)
        
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
                   
    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'params': self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))
    # Compute loss and gradient for a minibatch of data.
    def loss(self, X, y=None):
        scores = None
        inter, cache_LR = Linear_ReLU.forward(torch.tensor(X), self.params['W1'], self.params['b1'])
        scores, cache_L = Linear.forward(inter, self.params['W2'], self.params['b2'])
      
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        loss_unreg, grad_Softmax = softmax_loss(scores,y)
        
        W1_sq = torch.pow(self.params['W1'], 2)
        W2_sq = torch.pow(self.params['W2'], 2)
        loss = loss_unreg + torch.mul(torch.add(torch.sum(W2_sq), torch.sum(W1_sq)), self.reg)
        
        # Get Linear_ReLU upstream gradient
        gradX2, gradW2, gradB2 = Linear.backward(grad_Softmax, cache_L)
        gradX1, gradW1, gradB1 = Linear_ReLU.backward(gradX2, cache_LR)
        grads['W1'] = gradW1 + 2 * self.reg * self.params['W1']
        grads['b1'] = gradB1 
        grads['W2'] = gradW2 + 2 * self.reg * self.params['W2']
        grads['b2'] = gradB2
      
        return loss, grads

# A fully-connected neural network with an arbitrary number of hidden layers, 
# ReLU nonlinearities, and a softmax loss function.
class FullyConnectedNet(object):
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
      
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.device = device
        
        for i in range (len(hidden_dims)):
            if (i == 0):
                self.params['W1'] = torch.normal(mean=0.0, std=weight_scale, size=(input_dim, hidden_dims[i]), dtype=dtype, device=self.device)
                self.params['b1'] = torch.zeros(hidden_dims[i], dtype=dtype, device=self.device)
            else:
                self.params['W{i}'.format(i=str(i + 1))] = torch.normal(mean=0.0, std=weight_scale, size=(hidden_dims[i - 1], hidden_dims[i]), dtype=dtype, device=self.device)
                self.params['b{i}'.format(i=str(i + 1))] = torch.zeros(hidden_dims[i], dtype=dtype, device=self.device)
        self.params['W{i}'.format(i=str(len(hidden_dims) + 1))] = torch.normal(mean=0.0, std=weight_scale, size=(hidden_dims[len(hidden_dims) - 1], num_classes), dtype=dtype, device=self.device)
        self.params['b{i}'.format(i=str(len(hidden_dims) + 1))] = torch.zeros(num_classes, dtype=dtype, device=self.device)

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    # Compute loss and gradient for the fully-connected net.
    def loss(self, X, y=None):
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        scores = None
        dtype = self.dtype
        num_layers = self.num_layers
        
        cache = []
        
        regular_weights = 0

        for i in range (num_layers - 1):
            X, cache_LR = Linear_ReLU.forward(torch.tensor(X), self.params['W{i}'.format(i=str(i + 1))], self.params['b{i}'.format(i=str(i + 1))])
            cache.append(cache_LR)
            regular_weights += torch.sum(torch.square(self.params['W{i}'.format(i=str(num_layers))]))
        scores, cache_L = Linear.forward(X, self.params['W{i}'.format(i=str(num_layers))], self.params['b{i}'.format(i=str(num_layers))])
      
        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        loss, grad_softmax = softmax_loss(scores, y)
        loss += 0.5 * self.reg * regular_weights

        gradX, gradW, gradB = Linear.backward(grad_softmax, cache_L)
        grads['W{i}'.format(i=num_layers)] = gradW + 0.5 * self.reg * self.params['W{i}'.format(i=num_layers)]
        grads['b{i}'.format(i=num_layers)] = gradB
        
        for i in range (num_layers-1):
            gradX, gradW, gradB = Linear_ReLU.backward(gradX, cache[num_layers - 2 - i])
            grads['W{i}'.format(i=num_layers - 1 - i)] = gradW + 0.5 * self.reg * self.params['W{i}'.format(i=num_layers - 1 - i)]
            grads['b{i}'.format(i=num_layers - 1 - i)] = gradB 

        return loss, grads




