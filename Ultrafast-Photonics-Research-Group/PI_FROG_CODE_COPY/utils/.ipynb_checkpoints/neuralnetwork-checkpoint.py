import tensorflow as tf
import numpy as np
import os
from custom_lbfgs import lbfgs, Struct

'''
class Sine(tf.keras.layers.Layer):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def call(self, x):
        return tf.math.sin(self.w0 * x)

class SIRENInitializer(tf.keras.initializers.Initializer):
    def __init__(self, w0=1.0, is_first=False):
        self.w0 = w0
        self.is_first = is_first

    def __call__(self, shape, dtype=None):
        fan_in = shape[0]
        if self.is_first:
            return tf.random.uniform(shape, 
                                     minval=-1 / fan_in, 
                                     maxval=1 / fan_in,
                                     dtype=dtype)
        else:
            # From SIREN paper: sqrt(6/fan_in)/w0
            limit = np.sqrt(6 / fan_in) / self.w0
            return tf.random.uniform(shape, 
                                     minval=-limit, 
                                     maxval=limit,
                                     dtype=dtype)

class NeuralNetwork(object):
    def __init__(self, hp, logger, ub, lb, w0=30.0):
        layers = hp["layers"]

        # Add nt_config like old code for LBFGS config
        self.nt_config = Struct()
        self.nt_config.learningRate = hp["nt_lr"]
        self.nt_config.maxIter = hp["nt_epochs"]
        self.nt_config.startIter = 0
        self.nt_config.nCorrection = hp["nt_ncorr"]
        self.nt_config.tolFun = 1.0 * np.finfo(float).eps

        # TensorFlow optimizer config
        self.tf_epochs = hp["tf_epochs"]
        self.tf_optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp["tf_lr"],
            beta_1=hp["tf_b1"],
            epsilon=hp["tf_eps"]
        )

        self.dtype = "float64"


        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))

        # First hidden layer with special initialization
        self.model.add(tf.keras.layers.Dense(
            layers[1],
            kernel_initializer=SIRENInitializer(w0=w0, is_first=True),
            activation=None
        ))
        self.model.add(Sine(w0=w0))

        # Additional hidden layers
        for width in layers[2:-1]:
            self.model.add(tf.keras.layers.Dense(
                width,
                kernel_initializer=SIRENInitializer(w0=w0),
                activation=None
            ))
            self.model.add(Sine(w0=w0))

        # Output layer
        self.model.add(tf.keras.layers.Dense(
            layers[-1],
            kernel_initializer='glorot_uniform',
            activation=None
        ))
'''       
class NeuralNetwork(object):
    def __init__(self, hp, logger, ub, lb):

        layers = hp["layers"]
        # Setting up the optimizers with the hyper-parameters
        self.nt_config = Struct()
        self.nt_config.learningRate = hp["nt_lr"]
        self.nt_config.maxIter = hp["nt_epochs"]
        self.nt_config.startIter = 0
        self.nt_config.nCorrection = hp["nt_ncorr"]
        self.nt_config.tolFun = 1.0 * np.finfo(float).eps
        self.tf_epochs = hp["tf_epochs"]
        self.tf_optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp["tf_lr"],
            beta_1=hp["tf_b1"],
            epsilon=hp["tf_eps"])

        self.dtype = "float64"
        # Descriptive Keras model
        tf.keras.backend.set_floatx(self.dtype)
        
        
        self.model = tf.keras.Sequential()
        
        self.model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        self.model.add(tf.keras.layers.Lambda(
            lambda X: 2.0*(X - lb)/(ub - lb) - 1.0))
        for width in layers[1:-1]:
            self.model.add(tf.keras.layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer="glorot_normal"))
        self.model.add(tf.keras.layers.Dense(
                layers[-1], activation=None,
                kernel_initializer="glorot_normal"))
        
        # Computing the sizes of weights/biases for future decomposition
        
        self.sizes_w = []
        self.sizes_b = []
        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))

        self.sizes_w = []
        self.sizes_b = []
        
        #in_dim = layers[0]
        #for out_dim in layers[1:]:
        #    self.sizes_w.append(in_dim * out_dim)
        #    self.sizes_b.append(out_dim)
        #    in_dim = out_dim

        self.logger = logger

    # Defining custom loss
    # @tf.function
    def loss(self, u, u_pred):
        return tf.reduce_mean(tf.square(u - u_pred))

    # @tf.function
    def grad(self, X, u):
        with tf.GradientTape() as tape:
            loss_value = self.loss(u, self.model(X))
        grads = tape.gradient(loss_value, self.wrap_training_variables())
        return loss_value, grads

    def wrap_training_variables(self):
        var = self.model.trainable_variables
        return var

    def get_params(self, numpy=False):
        return []
    
    def get_weights(self, convert_to_tensor=True, LayerInx = None):
        w = []
        if type(LayerInx) == type(None):
            LayerInx = np.arange(1,len(self.model.layers[1:])+1)
        for i in LayerInx:
            layer = self.model.layers[int(i)]
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        if convert_to_tensor:
            w = self.tensor(w)
        return w
    '''
    def get_weights(self, convert_to_tensor=True, LayerInx=None):
        w = []
        if LayerInx is None:
            LayerInx = np.arange(len(self.model.layers))  # include all layers
    
        for i in LayerInx:
            layer = self.model.layers[i]
            if isinstance(layer, tf.keras.layers.Dense):
                weights, biases = layer.get_weights()
                w.extend(weights.flatten())
                w.extend(biases)
    
        if convert_to_tensor:
            w = tf.convert_to_tensor(w, dtype=tf.float64)
        return w

    '''
    def set_weights(self, w, LayerInx = None):
        for i, layer in enumerate(self.model.layers[1:]):
            if not layer.__class__.__name__ == 'Reshape':
                start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
                end_weights = sum(self.sizes_w[:i+1]) + sum(self.sizes_b[:i])
                weights = w[start_weights:end_weights]
                w_div = int(self.sizes_w[i] / self.sizes_b[i])
                weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
                biases = w[end_weights:end_weights + self.sizes_b[i]]
                weights_biases = [weights, biases]
                layer.set_weights(weights_biases)
    '''           
    def set_weights(self, w_flat, LayerInx=None):
        pointer = 0
        layer_idx = 0
    
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                w_size = self.sizes_w[layer_idx]
                b_size = self.sizes_b[layer_idx]
    
                weights = tf.reshape(w_flat[pointer:pointer + w_size],
                                     [int(w_size / b_size), b_size])
                pointer += w_size
    
                biases = w_flat[pointer:pointer + b_size]
                pointer += b_size
    
                layer.set_weights([weights.numpy(), biases.numpy()])
                layer_idx += 1

    '''
    def get_loss_and_flat_grad(self, X, u):
        def loss_and_flat_grad(w):
            with tf.GradientTape() as tape:
                self.set_weights(w)
                loss_value = self.loss(u, self.model(X))
            grad = tape.gradient(loss_value, self.wrap_training_variables())
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)
            return loss_value, grad_flat

        return loss_and_flat_grad

    def tf_optimization(self, X_u, u):
        self.logger.log_train_opt("Adam")
        for epoch in range(self.tf_epochs):
            loss_value = self.tf_optimization_step(X_u, u)
            self.logger.log_train_epoch(epoch+self.logger.get_total_epoch(), loss_value,self)

    # @tf.function
    def tf_optimization_step(self, X_u, u):
        loss_value, grads = self.grad(X_u, u)
        
        self.tf_optimizer.apply_gradients(
                zip(grads, self.wrap_training_variables()))
        return loss_value

    def nt_optimization(self, X_u, u):
        self.logger.log_train_opt("LBFGS")
        loss_and_flat_grad = self.get_loss_and_flat_grad(X_u, u)
        # tfp.optimizer.lbfgs_minimize(
        #   loss_and_flat_grad,
        #   initial_position=self.get_weights(),
        #   num_correction_pairs=nt_config.nCorrection,
        #   max_iterations=nt_config.maxIter,
        #   f_relative_tolerance=nt_config.tolFun,
        #   tolerance=nt_config.tolFun,
        #   parallel_iterations=6)
        self.nt_optimization_steps(loss_and_flat_grad)

    def nt_optimization_steps(self, loss_and_flat_grad):
        lbfgs(loss_and_flat_grad,
              self.get_weights(),
              self.nt_config,
              Struct(), 
              True,
              lambda epoch, loss, NN, is_iter:
              self.logger.log_train_epoch(epoch+self.logger.get_total_epoch(), loss, NN, custom = "", is_iter = is_iter),
              NeurNet = self)

    def fit(self, X_u, u):
        self.logger.log_train_start(self)

        # Creating the tensors
        X_u = self.tensor(X_u)
        u = self.tensor(u)

        # Optimizing
        self.tf_optimization(X_u, u)
        self.nt_optimization(X_u, u)

        self.logger.log_train_end(self.tf_epochs + self.nt_config.maxIter + self.logger.get_total_epoch(),self)

    def predict(self, X_star):
        u_pred = self.model(X_star)
        return u_pred.numpy()

    def summary(self):
        return self.model.summary()

    def tensor(self, X):
        return tf.convert_to_tensor(X, dtype=self.dtype)

    def load_latest_checkpoint(self,chk_point_num = 'last',basemodel = False):
        
        if basemodel: # Load file from basemodel 
            if chk_point_num == 'last':
                latest_chk = tf.train.latest_checkpoint(os.path.join(self.logger.SaveDir,'BaseModel','Checkpoints'))
            else:  
                latest_chk = os.path.join(os.path.join(self.logger.SaveDir,'BaseModel','Checkpoints',str(chk_point_num)))
        else: # Load file from PINN file
            if chk_point_num == 'last':
                latest_chk = tf.train.latest_checkpoint(self.logger.Checkpoint_dir)
            else:  
                latest_chk = os.path.join(self.logger.Checkpoint_dir,str(chk_point_num))
            
        self.model.load_weights(latest_chk)
