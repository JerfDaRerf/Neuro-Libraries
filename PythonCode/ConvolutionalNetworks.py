"""
Implements convolutional networks in PyTorch.
"""
import torch

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
    # linear units (ReLUs).
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

#Uses the Adam update rule, which incorporates moving averages of both the
# gradient and its square and a bias correction term.
def adam(w, dw, config=None):
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', torch.zeros_like(w))
  config.setdefault('v', torch.zeros_like(w))
  config.setdefault('t', 0)

  next_w = None
  config['t'] += 1
  config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dw
  mt = config['m'] / (1-config['beta1']**config['t'])
  config['v'] = config['beta2']*config['v'] + (1-config['beta2'])*(dw*dw)
  vc = config['v'] / (1-(config['beta2']**config['t']))
  w = w - (config['learning_rate'] * mt)/ (torch.sqrt(vc) + config['epsilon'])
  next_w = w

  return next_w, config

class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        out = None

        # Pad input x
        x_padded = torch.nn.functional.pad(x, (conv_param['pad'], conv_param['pad'], conv_param['pad'], conv_param['pad']))
    
        # Compute output shape
        H_final = 1 + int((x.shape[2] + 2 * conv_param['pad'] - w.shape[2]) / conv_param['stride'])
        W_final = 1 + int((x.shape[3] + 2 * conv_param['pad'] - w.shape[3]) / conv_param['stride'])
    
        # Initialize output tensor
        out = torch.zeros((x.shape[0], w.shape[0], H_final, W_final), device=x.device, dtype=x.dtype)
    
        # Perform convolution
        for N in range(x.shape[0]):  # Iterate over batch dimension
            for F in range(w.shape[0]):  # Iterate over filter dimension
                for h in range(H_final):  # Iterate over output height
                    for width in range(W_final):  # Iterate over output width
                        # Compute the convolution at each spatial location
                        out[N, F, h, width] = torch.sum(x_padded[N, :, h*conv_param['stride']:(h*conv_param['stride'] + w.shape[2]), width*conv_param['stride']:(width*conv_param['stride'] + w.shape[3])] * w[F]) + b[F]

        cache = (x, w, b, conv_param)
        return out, cache


    @staticmethod
    def backward(dout, cache):
       
        dx, dw, db = None, None, None
        x, w, b, conv_param = cache

        top_pad = torch.zeros((x.shape[0], x.shape[1], 1, x.shape[3]), device=x.device)
        side_pad = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + conv_param['pad'] * 2, 1), device=x.device)
        x_pd = torch.clone(x)
        for i in range(conv_param['pad']):
          x_pd = torch.cat((top_pad, x_pd, top_pad), dim = 2)

        for i in range(conv_param['pad']):
          x_pd = torch.cat((side_pad, x_pd, side_pad), dim = 3)
        
        H_final = 1 + (int) ((x.shape[2] + 2 * conv_param['pad'] - w.shape[2]) / conv_param['stride'])
        W_final = 1 + (int) ((x.shape[3] + 2 * conv_param['pad'] - w.shape[3]) / conv_param['stride'])
        
        dx = torch.zeros_like(x_pd, device = dout.device)        
        dw = torch.zeros_like(w, device = dout.device)
        db = torch.sum(dout, dim=(0,2,3))

        for N in range(x_pd.shape[0]):
          for F in range(w.shape[0]):
            for h in range(H_final):
              for width in range(W_final):
                filtered_X = x_pd[N, :, h*conv_param['stride']:(h*conv_param['stride'] + w.shape[2]), width*conv_param['stride']:(width*conv_param['stride'] + w.shape[3])]
                dx[N,:,h*conv_param['stride']:(h*conv_param['stride'] + w.shape[2]),width*conv_param['stride']:(width*conv_param['stride'] + w.shape[3])] += dout[N,F,h,width] * w[F]
                dw[F] += dout[N,F,h,width] * filtered_X
        dx = dx[:,:, conv_param['pad']:conv_param['pad'] + x.shape[2], conv_param['pad']:conv_param['pad'] + x.shape[3]]
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
      
        out = None
        H_final = 1 + (int) ((x.shape[2] - pool_param['pool_height']) / pool_param['stride'])
        W_final = 1 + (int) ((x.shape[3] - pool_param['pool_width']) / pool_param['stride'])
        out = torch.zeros((x.shape[0], x.shape[1], H_final, W_final), device=x.device, dtype = x.dtype)

        for N in range(x.shape[0]):
          for C in range(x.shape[1]):
            for h in range(H_final):
              for w in range(W_final):
                out[N, C, h, w] = torch.max(x[N, C, h*pool_param['stride']:(h*pool_param['stride'] + pool_param['pool_height']), w*pool_param['stride']:(w*pool_param['stride'] + pool_param['pool_width'])])
    
        out = out.cuda()
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        dx = None
        x, pool_param = cache
        H_final = 1 + (int) ((x.shape[2] - pool_param['pool_height']) / pool_param['stride'])
        W_final = 1 + (int) ((x.shape[3] - pool_param['pool_width']) / pool_param['stride'])
        out = torch.zeros((x.shape[0], x.shape[1], H_final, W_final), device=x.device, dtype = x.dtype)

        dx = torch.zeros_like(x, device = dout.device)        

        for N in range(x.shape[0]):
          for C in range(x.shape[1]):
            for h in range(H_final):
              for w in range(W_final):
                prePool = x[N, C, h*pool_param['stride']:(h*pool_param['stride'] + pool_param['pool_height']), w*pool_param['stride']:(w*pool_param['stride'] + pool_param['pool_width'])]
                maxPos = (torch.max(prePool) == prePool).nonzero()
                hpos = maxPos[0][0].item()
                wpos = maxPos[0][1].item()
                dx[N, C, h*pool_param['stride']:(h*pool_param['stride'] + pool_param['pool_height']), w*pool_param['stride']:(w*pool_param['stride'] + pool_param['pool_width'])][hpos, wpos] = dout[N, C, h, w]
       
        return dx

# A three-layer convolutional network with the following architecture: 
# conv - relu - 2x2 max pool - linear - relu - linear - softmax
class ThreeLayerConvNet(object):
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        self.device = device
        conv_pad = (filter_size - 1) // 2
        conv_stride = 1
        pool_height = 2
        pool_width = 2
        
        # Find size of each layer
        
        # (F, C, HH, WW) - where F is the number of filters, C is the number of input channels,
        # HH and WW are the height and width of the filters respectively
        W1 = torch.normal(mean=0.0, std=weight_scale, size=(num_filters, input_dims[0], filter_size, filter_size), dtype=dtype, device=self.device)
        
        # (F,) - one bias per filter
        b1 = torch.zeros(num_filters, dtype=dtype, device=self.device)
        
        # Calculate the size of the output of the convolutional layer
        # This will be used to determine the size of the input to the hidden linear layer
        
        # Calculate the height and width of the output after convolution and pooling
        H_prime = 1 + (input_dims[1] + 2 * conv_pad - filter_size) / conv_stride 
        W_prime = 1 + (input_dims[2] + 2 * conv_pad - filter_size) / conv_stride
        
        # Calculate the number of features (D) after pooling
        D = int(num_filters * (H_prime / pool_height)  * (W_prime / pool_width))
        
        # Initialize weights for the hidden linear layer (fully connected layer)
        # (D, M) - where D is the number of features from the previous layer and M is the number of hidden units
        W2 = torch.normal(mean=0.0, std=weight_scale, size=(D , int(hidden_dim)), dtype=dtype, device=self.device)
    
        # Initialize biases for the hidden linear layer
        # (M,) - one bias per hidden unit
        b2 = torch.zeros(hidden_dim, dtype=dtype, device=self.device)
        
        # Initialize weights for the output linear layer
        # (M, num_classes) - where M is the number of hidden units and num_classes is the number of output classes
        W3 = torch.normal(mean=0.0, std=weight_scale, size=(hidden_dim, num_classes), dtype=dtype, device=self.device)
        
        # Initialize biases for the output linear layer
        # (num_classes,) - one bias per output class
        b3 = torch.zeros(num_classes, dtype=dtype, device=self.device)
        
        # Store weights and biases in the dictionary self.params
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        self.params['W3'] = W3
        self.params['b3'] = b3

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
       
        # Forward pass for 3 layer conv net

        # 1. Conv_ReLU_Pool.forward(x, w, b, conv_param, pool_param)
        out_CRP, cache_CRP = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        
        # 2. Linear_ReLU.forward(x, w, b)
        out_LR, cache_LR = Linear_ReLU.forward(out_CRP, W2, b2)
        
        # 3. Linear.forward(x, w, b)
        scores, cache_L = Linear.forward(out_LR, W3, b3)

        if y is None:
            return scores

        loss, grads = 0.0, {}
        # backward pass for three-layer convolutional net

        loss_unreg, gradS = softmax_loss(scores, y)
        
        W1_2 = torch.pow(self.params['W1'], 2)
        W2_2 = torch.pow(self.params['W2'], 2)
        W3_2 = torch.pow(self.params['W3'], 2)
        inter_add = torch.add(torch.sum(W2_2), torch.sum(W1_2))
        loss = loss_unreg + torch.mul(torch.add(inter_add, torch.sum(W3_2)), self.reg)
        
        # Get Linear_ReLU upstream gradient
        gradX3, gradW3, gradB3 = Linear.backward(gradS, cache_L)
        gradX2, gradW2, gradB2 = Linear_ReLU.backward(gradX3, cache_LR)
        gradX1, gradW1, gradB1 = Conv_ReLU_Pool.backward(gradX2, cache_CRP)
        grads['W1'] = gradW1 + 2 * self.reg * self.params['W1']
        grads['b1'] = gradB1 
        grads['W2'] = gradW2 + 2 * self.reg * self.params['W2']
        grads['b2'] = gradB2
        grads['W3'] = gradW3 + 2 * self.reg * self.params['W3']
        grads['b3'] = gradB3

        return loss, grads

# A convolutional neural network with an arbitrary number of convolutional layers in VGG-Net style.
# The network will have the following architecture: {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear
class DeepConvNet(object):
    
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
       
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        self.device = device
        conv_pad = 1
        conv_stride = 1
        filter_size = 3
        pool_height = 2
        pool_width = 2
        pool_stride = 2
        
        currHeight = input_dims[1]
        currWidth = input_dims[2]
        for i in range (len(num_filters)):
            if (weight_scale == "kaiming"):
                if (i == 0):
                    # Update H, W for conv layer    
                    currHeight = 1 + (currHeight + 2 * conv_pad - filter_size) / conv_stride
                    currWidth = 1 + (currWidth + 2 * conv_pad - filter_size) / conv_stride

                    # (Din, Dout)
                    self.params['W1'] = kaiming_initializer(int(input_dims[0]), int(num_filters[i]), K = filter_size, dtype=dtype, device=self.device)
                    # (F,)
                    self.params['b1'] = torch.zeros(num_filters[i], dtype=dtype, device=self.device)
                    
                    if (batchnorm == True):
                        # Initialize gamma (scale), beta (shift)
                        self.params['scale{i}'.format(i=str(i + 1))] = torch.ones(int(num_filters[i]), dtype=dtype, device=self.device)
                        self.params['shift{i}'.format(i=str(i + 1))] = torch.zeros(int(num_filters[i]), dtype=dtype, device=self.device)
                        
                    # Update H, W for pool layer    
                    if (max_pools.count(i) > 0):
                        currHeight = 1 + (currHeight - pool_height) / pool_stride
                        currWidth = 1 + (currWidth - pool_width) / pool_stride            
                else:
                    # Update H, W for conv layer    
                    currWidth = 1 + (currHeight + 2 * conv_pad - filter_size) / conv_stride
                    currWidth = 1 + (currWidth + 2 * conv_pad - filter_size) / conv_stride

                    # (Din, Dout)
                    self.params['W{i}'.format(i=str(i + 1))] = kaiming_initializer(int(num_filters[i - 1]), int(num_filters[i]), K = filter_size, dtype=dtype, device=self.device)
                    # (F,)
                    self.params['b{i}'.format(i=str(i + 1))] = torch.zeros(num_filters[i], dtype=dtype, device=self.device)

                    if (batchnorm == True):             
                        # Initialize gamma (scale)d, beta (shift)
                        self.params['scale{i}'.format(i=str(i + 1))] = torch.ones(int(num_filters[i]), dtype=dtype, device=self.device)
                        self.params['shift{i}'.format(i=str(i + 1))] = torch.zeros(int(num_filters[i]), dtype=dtype, device=self.device)
                    
                    # Update H, W for pool layer    
                    if (max_pools.count(i) > 0):
                        currHeight = 1 + (currHeight - pool_height) / pool_stride
                        currWidth = 1 + (currWidth - pool_width) / pool_stride
            # No kaiming
            else:
                if (i == 0):
                    # (F, C, HH, WW)
                    self.params['W1'] = torch.normal(mean=0.0, std=weight_scale, size=(num_filters[i], input_dims[0], filter_size, filter_size), dtype=dtype, device=self.device)
                    # (F,)
                    self.params['b1'] = torch.zeros(num_filters[i], dtype=dtype, device=self.device)
                      
                    # Update H, W for conv layer    
                    currHeight = 1 + (currHeight + 2 * conv_pad - filter_size) / conv_stride
                    currWidth = 1 + (currWidth + 2 * conv_pad - filter_size) / conv_stride
                
                    if (batchnorm == True):
                        # Initialize gamma (scale), beta (shift)
                        self.params['scale{i}'.format(i=str(i + 1))] = torch.ones(int(num_filters[i]), dtype=dtype, device=self.device)
                        self.params['shift{i}'.format(i=str(i + 1))] = torch.zeros(int(num_filters[i]), dtype=dtype, device=self.device)
                        
                    # Update H, W for pool layer    
                    if (max_pools.count(i) > 0):
                        currHeight = 1 + (currHeight - pool_height) / pool_stride
                        currWidth = 1 + (currWidth - pool_width) / pool_stride
                else:
                    # (F, C, HH, WW)
                    self.params['W{i}'.format(i=str(i + 1))] = torch.normal(mean=0.0, std=weight_scale, size=(num_filters[i], num_filters[i-1], filter_size, filter_size), dtype=dtype, device=self.device)
                    # (F,)
                    self.params['b{i}'.format(i=str(i + 1))] = torch.zeros(num_filters[i], dtype=dtype, device=self.device)
                    
                    # Update H, W for conv layer    
                    currHeight = 1 + (currHeight + 2 * conv_pad - filter_size) / conv_stride
                    currWidth = 1 + (currWidth + 2 * conv_pad - filter_size) / conv_stride

                    if (batchnorm == True):             
                        # Initialize gamma (scale)d, beta (shift)
                        self.params['scale{i}'.format(i=str(i + 1))] = torch.ones(int(num_filters[i]), dtype=dtype, device=self.device)
                        self.params['shift{i}'.format(i=str(i + 1))] = torch.zeros(int(num_filters[i]), dtype=dtype, device=self.device)
                    
                    # Update H, W for pool layer    
                    if (max_pools.count(i) > 0):
                        currHeight = 1 + (currHeight - pool_height) / pool_stride
                        currWidth = 1 + (currWidth - pool_width) / pool_stride
        
        if weight_scale == "kaiming":
            # Kaiming initialization for the final linear layer
            # Initialize weights using Kaiming initialization
            self.params['W{i}'.format(i=str(len(num_filters) + 1))] = kaiming_initializer(int(num_filters[len(num_filters) - 1] * currHeight * currWidth), num_classes, dtype=dtype, device=self.device)
            
            # Initialize biases to zeros
            self.params['b{i}'.format(i=str(len(num_filters) + 1))] = torch.zeros(num_classes, dtype=dtype, device=self.device)        
        else:
            # Standard Gaussian initialization for the final linear layer
            # Initialize weights from a Gaussian distribution with zero mean and standard deviation equal to weight_scale
            self.params['W{i}'.format(i=str(len(num_filters) + 1))] = torch.normal(mean=0.0, std=weight_scale, size=(int(num_filters[len(num_filters) - 1] * currHeight * currWidth), num_classes), dtype=dtype, device=self.device)
            
            # Initialize biases to zeros
            self.params['b{i}'.format(i=str(len(num_filters) + 1))] = torch.zeros(num_classes, dtype=dtype, device=self.device)

        # With batch normalization we need to keep track of running means and variances
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):

        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
       
        # Forward pass for the DeepConvNet, computing the class scores for X and storing them 
        dtype = self.dtype
        num_layers = self.num_layers
        max_pools = self.max_pools
        batchnorm = self.batchnorm
        bn_param = self.bn_params
        
        cache = []
        
        regular_weights = 0

        # Loop through each layer in the network
        for i in range(num_layers - 1):
            # Apply convolution, batch normalization (if enabled), and ReLU activation
            if batchnorm:
                if i in max_pools:
                    # If the layer follows a max pooling layer, apply convolution, batch normalization, ReLU, and max pooling
                    X, cache_LR = Conv_BatchNorm_ReLU_Pool.forward(X.clone().detach(), self.params['W{i}'.format(i=str(i + 1))], self.params['b{i}'.format(i=str(i + 1))], self.params['scale{i}'.format(i=str(i + 1))], self.params['shift{i}'.format(i=str(i + 1))], conv_param, bn_param[i], pool_param)
                else:
                    # If the layer does not follow a max pooling layer, apply convolution and batch normalization
                    X, cache_LR = Conv_BatchNorm_ReLU.forward(X.clone().detach(), self.params['W{i}'.format(i=str(i + 1))], self.params['b{i}'.format(i=str(i + 1))], self.params['scale{i}'.format(i=str(i + 1))], self.params['shift{i}'.format(i=str(i + 1))], conv_param, bn_param[i])
            else:
                if i in max_pools:
                    # If the layer follows a max pooling layer, apply convolution, ReLU, and max pooling
                    X, cache_LR = Conv_ReLU_Pool.forward(X.clone().detach(), self.params['W{i}'.format(i=str(i + 1))], self.params['b{i}'.format(i=str(i + 1))], conv_param, pool_param)
                else:
                    # If the layer does not follow a max pooling layer, apply convolution and ReLU
                    X, cache_LR = Conv_ReLU.forward(X.clone().detach(), self.params['W{i}'.format(i=str(i + 1))], self.params['b{i}'.format(i=str(i + 1))], conv_param)
            
            # Append the computed cache for this layer to the cache list
            cache.append(cache_LR)
            
            # Accumulate the regularization term for weights
            regular_weights += torch.sum(torch.square(self.params['W{i}'.format(i=str(num_layers))]))
        
        # Flatten the output of the last convolutional layer
        X = X.contiguous()
        scores, cache_L = Linear.forward(X, self.params['W{i}'.format(i=str(num_layers))], self.params['b{i}'.format(i=str(num_layers))])

        if y is None:
            return scores

        loss, grads = 0, {}
        # Backward pass for the DeepConvNet, storing the loss and gradients 
        loss, gradS = softmax_loss(scores, y)
        loss += 0.5 * self.reg * torch.sum(torch.square(self.params['W{i}'.format(i=num_layers)]))
        gradX, gradW, gradB = Linear.backward(gradS, cache_L)    
        grads['W{i}'.format(i=num_layers)] = gradW + self.reg * self.params['W{i}'.format(i=num_layers)]
        grads['b{i}'.format(i=num_layers)] = gradB 
        for i in range (num_layers-1):
            # Cases: batchnorm (Y/N), Pool (Y/N)
            loss += 0.5 * self.reg * torch.sum(torch.square(self.params['W{i}'.format(i=num_layers - 1 - i)]))
            if(batchnorm):
                if(max_pools.count(num_layers - 2 - i) > 0):
                    gradX, gradW, gradB, gradGamma, gradBeta = Conv_BatchNorm_ReLU_Pool.backward(gradX, cache[num_layers - 2 - i])
                    grads['W{i}'.format(i=num_layers - 1 - i)] = gradW + self.reg * self.params['W{i}'.format(i=num_layers - 1 - i)]
                    grads['b{i}'.format(i=num_layers - 1 - i)] = gradB
                    grads['scale{i}'.format(i=num_layers - 1 - i)] = gradGamma
                    grads['shift{i}'.format(i=num_layers - 1 - i)] = gradBeta
                else:
                    gradX, gradW, gradB, gradGamma, gradBeta = Conv_BatchNorm_ReLU.backward(gradX, cache[num_layers - 2 - i])
                    grads['W{i}'.format(i=num_layers - 1 - i)] = gradW + self.reg * self.params['W{i}'.format(i=num_layers - 1 - i)]
                    grads['b{i}'.format(i=num_layers - 1 - i)] = gradB
                    grads['scale{i}'.format(i=num_layers - 1 - i)] = gradGamma
                    grads['shift{i}'.format(i=num_layers - 1 - i)] = gradBeta     
            else:
                if(max_pools.count(num_layers - 2 - i) > 0):
                    gradX, gradW, gradB = Conv_ReLU_Pool.backward(gradX, cache[num_layers - 2 - i])
                    grads['W{i}'.format(i=num_layers - 1 - i)] = gradW + self.reg * self.params['W{i}'.format(i=num_layers - 1 - i)]
                    grads['b{i}'.format(i=num_layers - 1 - i)] = gradB
                else:
                    gradX, gradW, gradB = Conv_ReLU.backward(gradX, cache[num_layers - 2 - i])
                    grads['W{i}'.format(i=num_layers - 1 - i)] = gradW + self.reg * self.params['W{i}'.format(i=num_layers - 1 - i)]
                    grads['b{i}'.format(i=num_layers - 1 - i)] = gradB                       

        return loss, grads
# Kaiming initialization for linear and convolution layers.
def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        weight = torch.randn(size=(Din, Dout), dtype=dtype, device=device) * ((gain / Din) ** 0.5)
    else:
        weight = torch.randn(size=(Dout, Din, K, K), dtype=dtype, device=device) * ((gain / (Din * K * K)) ** 0.5)
    return weight


class BatchNorm(object):

    # Forward pass for batch normalization.
    @staticmethod
    def forward(x, gamma, beta, bn_param):
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            # Replace "pass" statement with your code
            
            sample_mean_new = torch.mean(x, dim=0)
            sample_var_new = torch.var(x, correction=0, dim=0)
            
            running_mean_new = momentum * running_mean + (1 - momentum) * sample_mean_new
            running_var_new = momentum * running_var + (1 - momentum) * sample_var_new
            
            centered_new = x - sample_mean_new
            standardDev_new = torch.sqrt(sample_var_new + eps)
            x_normalized = (x - sample_mean_new) / torch.sqrt(sample_var_new + eps)
            out = gamma * x_normalized + beta
            cache = (x_normalized, centered_new, standardDev_new, mode, gamma, beta)
        elif mode == 'test':
            # Replace "pass" statement with your code
            x_normalized = (x - running_mean) / torch.sqrt(running_var + eps)
            out = gamma * x_normalized + beta
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    # Backward pass for batch normalization.
    @staticmethod
    def backward(dout, cache):
        dx, dgamma, dbeta = None, None, None

        # Backward pass for batch normalization.
        x_normalized, centered, standardDev, mode, gamma, beta = cache
        dgamma = torch.sum(dout * x_normalized, dim=0)
        dbeta = torch.sum(dout, dim=0)
        dx_inter = dout * gamma
        dx = (1 / dout.shape[0] / standardDev) * (
                    dout.shape[0] * dx_inter - torch.sum(dx_inter, dim=0) - x_normalized * torch.sum(
                dx_inter * x_normalized, dim=0))

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    # Computes the forward pass for spatial batch normalization.
    @staticmethod
    def forward(x, gamma, beta, bn_param):
        out, cache = None, None

        x_reshaped = torch.permute(x, (0, 2, 3, 1)).reshape(-1, x.shape[1])
        out, cache = BatchNorm.forward(x_reshaped, gamma, beta, bn_param)
        out = torch.permute(out.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[1]), (0, 3, 1, 2))

        return out, cache

    # Computes the backward pass for spatial batch normalization.
    @staticmethod
    def backward(dout, cache):
        dx, dgamma, dbeta = None, None, None

        dout_reshaped = torch.permute(dout, (0, 2, 3, 1)).reshape(-1, dout.shape[1])
        dx, dgamma, dbeta = BatchNorm.backward(dout_reshaped, cache)
        dx = torch.permute(dx.reshape(dout.shape[0], dout.shape[2], dout.shape[3], dout.shape[1]), (0, 3, 1, 2))

        return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    # A convenience layer that performs a convolution followed by a ReLU.
    @staticmethod
    def forward(x, w, b, conv_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):

        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
       
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
      
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta
