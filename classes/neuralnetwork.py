import os
import pickle
import tensorflow as tf

def reload_model(model_dir, inputshape):
    """Load a model.
    """
    model_name = os.path.basename(model_dir)
    weights_path = os.path.abspath(os.path.join(model_dir, model_name))
    config_path = os.path.abspath(os.path.join(model_dir, model_name + ".config"))
    with open(config_path, 'rb') as filehandle:
        config = pickle.load(filehandle)
    model = ValueModel(**config)
    model.build_graph(belief_map_shape_nobatch=inputshape)
    model.load_weights(weights_path)
    return model


class ValueModel(tf.keras.Model):
    """
    ValueModel developed with DRL

    From Loisy article: 
    "DQN is a model-free method. However, this problem is model-based because the probability of transitioning
    from a belief state s to a successor belief state s' is known exactly. This allows us to work directly with the value
    function v_hat, and to perform a full backup (to compute the sum over s') rather than a sample backup (based on a single
    successor belief state randomly sampled.)

    This network is identical to DQN except that the network is trained to approximate the value function rather than the 
    action-value function, and uses full backups rather than sample ones."

    Parameters
    ----------
    model_name: string
        'online_network' or 'target_network'
    Ndim: scalar (int)
        Number of space dimensions (2D or 3D) for the problem
    FC_layers: scalar (int)
        Number of hidden fully-connected layers
    FC_units: scalar (int) or list (int)
        Units per layer
    CNN_layers: scalar (int)
        Number of hidden CNN layers
    CNN_filters: scalar (int) or list (int)
        Number of output filters in the convolutions
    CNN_kernel_size: scalar (int) or list (int)
        Height and width of the 2D convolution window
        NOTE only implemented square kernels
    regularization_factor: scalar (float), optional
        Factor for regularization losses (default = 0.0)
    """

    def __init__(self, 
                 model_name,
                 Ndim,
                 FC_layers,
                 FC_units,
                 CNN_layers,
                 CNN_filters,
                 CNN_kernel_size,
                 pooling,
                 pooling_kernel_size,
                 pooling_strides,
                 regularization_factor,
                 loss_function='mean_squared_error',
                 ):

        """
        Initialize the neural network
        """
        super(ValueModel, self).__init__() # TODO in otto. Needed?

        # Set the model name: online_network or target_network
        self.model_name = model_name

        # Set mandatory domain variables
        self.Ndim = Ndim

        # Loss function TODO implement other options?
        # ChatGPT for discrete action spaces, the mean squared error or Huber loss are often used
        if loss_function == 'mean_squared_error': 
            self.loss_function = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE) 
        elif loss_function == 'huber':
            self.loss_function = tf.keras.losses.Huber()
        else: # if you reload a model
            self.loss_function = loss_function
        # print(f"loss_function = {loss_function}")

        regularizer = tf.keras.regularizers.l2(regularization_factor) 

        # Flattening 
        self.flatten = tf.keras.layers.Flatten()

        # CNN layers
        if CNN_layers > 0:

            if isinstance(CNN_filters, int):
                CNN_filters = tuple([CNN_filters] * CNN_layers)
            else:
                CNN_filters = tuple(CNN_filters)
            if len(CNN_filters) != CNN_layers:
                raise Exception ("User must provide the number of filters for each CNN layer or provide a single value to be applied to each CNN layer")
            
            if isinstance(CNN_kernel_size, int):
                CNN_kernel_size = tuple([CNN_kernel_size] * CNN_layers)
            else:
                CNN_kernel_size = tuple(CNN_kernel_size)
            if len(CNN_kernel_size) != CNN_layers:
                raise Exception ("User must provide the square kernel size for each CNN layer or provide a single value to be applied to each CNN layer")

            if isinstance(pooling_kernel_size, int):
                pooling_kernel_size = tuple([pooling_kernel_size] * (CNN_layers-1))
            else:
                pooling_kernel_size = tuple(pooling_kernel_size)
            if len(pooling_kernel_size) != CNN_layers-1:
                raise Exception ("User must provide the square kernel size for each in between CNN layer or provide a single value to be applied to each in between CNN layer")
            
            if isinstance(pooling_strides, int):
                pooling_strides = tuple([pooling_strides] * (CNN_layers-1))
            else:
                pooling_strides = tuple(pooling_strides)
            if len(pooling_strides) != CNN_layers-1:
                raise Exception ("User must provide the strides for each pooling layer or provide a single value to be applied to each pooling layer")

            self.CNN_block = []
            for i in range(CNN_layers):
                CNN_layer = tf.keras.layers.Conv2D(
                    filters = CNN_filters[i],
                    kernel_size = CNN_kernel_size[i], # NOTE only square kernels possible for now
                    activation = 'relu',
                    padding = 'same', # 'same' = padding (maintains the input width and height), default is 'valid' = no padding.
                    kernel_regularizer = regularizer,
                )
                self.CNN_block.append(CNN_layer)
                # AveragePooling in between conv layers
                if pooling and i != CNN_layers-1:
                    pool_layer = tf.keras.layers.AveragePooling2D(
                        pool_size = pooling_kernel_size[i],
                        padding = 'same',
                        strides = pooling_strides[i],
                    )
                    self.CNN_block.append(pool_layer)
        else:
            self.CNN_block = None

        # Fully connected layers
        if FC_layers > 0:
            if isinstance(FC_units, int): 
                FC_units = tuple([FC_units] * FC_layers)
            else:
                FC_units = tuple(FC_units)
            if len(FC_units) != FC_layers:
                raise Exception ("User must provide the number of units for each dense layer or provide a single value to be applied to each dense layer")
            
            self.FC_block = []
            for i in range(FC_layers):
                dense_layer = tf.keras.layers.Dense(
                    units = FC_units[i], # dimensionality of the output space
                    activation = 'relu', # activation function
                    kernel_initializer = tf.keras.initializers.HeUniform(), # initializer for the kernel weights matrix
                    # NOTE Watch out! 2 types of regularizers!
                    activity_regularizer = regularizer, # regularizer function applied to the output of the layer (it's "activation")
                )
                self.FC_block.append(dense_layer)
        else:
            self.FC_block = None
        
        # Last linear layer: In our case the output is v(s), not q(s,a)!
        self.final_layer = tf.keras.layers.Dense(
            units = 1,
            activation = None, # for regression tasks activation is None
            kernel_initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1),
            kernel_constraint = tf.keras.constraints.non_neg(), # constraint function applied to the kernel weights matrix
            bias_constraint = tf.keras.constraints.non_neg(), # constraint function applied to the bias vector
        )

        self.config = {"model_name": self.model_name,
                       "Ndim": self.Ndim,
                       "FC_layers": FC_layers,
                       "FC_units": FC_units,
                       "CNN_layers": CNN_layers,
                       "CNN_filters": CNN_filters,
                       "CNN_kernel_size": CNN_kernel_size,
                       "pooling": pooling,
                       "pooling_kernel_size": pooling_kernel_size,
                       "pooling_strides": pooling_strides,
                       "regularization_factor": regularization_factor,
                       "loss_function": self.loss_function,
                       }

    def build_graph(self, belief_map_shape_nobatch):
        """
        Builds the neural network model. 
        We use this function instead of tf.keras.model.build() so that a call to 
        tf.keras.model.summary() gives shape information. 

        Arguments
        ---------
        belief_map_shape_nobatch: tuple (int) 
            Shape of input to the neural network given by env.NN_input_shape
            This is: tuple([self.Nflux] + [2 * self.Ngrid - 1] * self.Ndim)
        """

        # Shape of the input to the neural network. This is the belief map centered around the agent.
        belief_map_shape_nobatch = tuple(belief_map_shape_nobatch)
        belief_map_shape_withbatch = tuple([1] + list(belief_map_shape_nobatch))
        print(f"belief_map_shape_nobatch = {belief_map_shape_nobatch} and belief_map_shape_withbatch = {belief_map_shape_withbatch}")

        self.build(belief_map_shape_withbatch)
        belief_map = tf.keras.Input(shape=belief_map_shape_nobatch)
        _ = self.call(belief_map)

    def call(self, x, training=False):
        """
        Call the ValueModel (neural network)
        
        Parameters
        ---------
        x (belief_map): ndarray or tf.tensor with shape (batch_size, NN_input_shape)
            array containing a batch of belief maps
        training: bool, optional 
            whether this call is done during training (training=True) as opposed to evaluation (training=False), default=False

        Returns
        -------
        x: tf.tensor with shape(batch_size, 1)
            array containing a batch of values
        """

        # Reshape belief_map from channels_first format (batch_size, channels, height, width)
        # to channels_last format (batch_size, height, width, channels)
        # channels_first format is not supported by tensorflow on CPU.

        if self.Ndim == 1:
            x = tf.expand_dims(x, axis=-1) # adds new axis for conv2D
        x = tf.transpose(x, [0, 2, 3, 1]) # from batch_size, channels (fluxes), height, width to batch_size, height, width, channels

        # Forward pass of belief_map through CNN
        if self.CNN_block is not None:
            for i in range(len(self.CNN_block)):
                x = self.CNN_block[i](x, training=training)

        # Flatten
        x = self.flatten(x)

        # Forward pass of belief map through Dense layers
        if self.FC_block is not None:
            for i in range(len(self.FC_block)):
                x = self.FC_block[i](x, training=training)

        # Final layer
        x = self.final_layer(x)

        return x # (batch_size, 1)
    
    @tf.function
    def train_step(self, x, y):
        """
        Perform a training step.

        1. Compute predictions
        2. Compute loss
        3. Compute gradients
        4. Update weights
        5. Compute total loss

        Arguments
        ---------
        x (belief_map): tf.tensor with shape (batch_size, NN_input_shape)
            array containing a batch of belief maps centered around agent inputs
        y: tf.tensor with shape (batch_size, 1)
            Batch of target values

        Returns
        -------
        loss: tf.tensor with shape ()
            Total loss
        """

        with tf.GradientTape() as tape:

            # 1. Compute predictions
            y_pred = self(x, training=True) # forward pass

            # 2. Compute loss
            loss_err = self.loss_function(y, y_pred) # compute loss
            loss_reg = tf.math.reduce_sum(self.losses) # adding the regularization losses
            loss = tf.math.add(loss_err, loss_reg)

        # 3. Compute gradients
        trainable_vars = self.trainable_variables # trainable_vars = weights p.65 
        gradients = tape.gradient(loss, trainable_vars)

        # 4. Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # 5. Compute total loss
        loss = tf.math.reduce_mean(loss)  # p.65 gradients are computed from the average_loss
        # print(f"loss = {loss}")

        return loss
    
    @tf.function
    def test_step(self, x, y):
        """
        Perform a test step. NOTE not used?

        1. Compute predictions
        2. Compute loss
                Skip step 3 and 4 of train_step
        5. Compute total loss

        Arguments
        ---------
        x (belief_map): tf.tensor with shape (batch_size, NN_input_shape)
            array containing a batch of belief maps centered around agent inputs
        y: tf.tensor with shape (batch_size, 1)
            Batch of target values

        Returns
        -------
        loss: tf.tensor with shape = ()
            Total loss
        """

        # 1. Compute predictions (NOTE without tape)
        y_pred = self(x, training=False) # forward pass

        # 2. Compute loss 
        loss_err = self.loss_function(y, y_pred)  # compute loss
        loss_reg = tf.math.reduce_sum(self.losses)  # adding the regularization losses
        loss = tf.math.add(loss_err, loss_reg)

        # 5. Compute total loss
        loss = tf.math.reduce_mean(loss)

        return loss
    
    def save_model(self, model_dir):
        """ 
        Save the model in the model_dir 
        """
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        model_name = os.path.basename(model_dir)
        weights_path = os.path.abspath(os.path.join(model_dir, model_name))
        self.save_weights(weights_path, save_format='h5') # this is a tensorflow function
        config_path = os.path.abspath(os.path.join(model_dir, model_name + ".config"))
        with open(config_path, 'wb') as filehandle: # pickle the config
            pickle.dump(self.config, filehandle)