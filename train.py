from keras import backend as K
from keras import layers

def margin_los(y_true, y_pred):
    """
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred-0.1))
    
    return K.mean(K.sum(L, 1))

class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        # L2 length which is the square root
        # of the sum of the capsule element
        return K.sqrt(K.sum(K.square(inputs), -1))

class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, d1, d2] by the max value in axis=1.
    Output shape: [None, d2]
    """
    def call(self, inputs, **kwargs):
        # user true label to select target capsule, shape=[batch_size, num_capsule]
        if type(inputs) is list: # true label is provided with shape = [batch_size, n_classes], i.e. one-hot encoding
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = inputs
            # Enlarge the range of values in x to make max(new_x)=1 and others <0
            x = (x - K.max(x, 1, True)) / K.epsilon() + 1
            mask = K.clip(x, 0, 1) # the max value in x clipped to 1 and other to 0

        # masked inputs, shape = [batch_size, dim_vector]
        inputs_masked = K.batch_dot(inputs, mask, [1, 1])
        return inputs_masked

def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector near 1 and a small vector near 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)/ K.sqrt(s_squared_norm)
    return scale * vectors

class DigiCap(layers.Layer):
    """
    The capsules layer.

    :param num_capsule: number of capsules in this layer
    :param dim_vector: dimension of the output vectors of the capsules in this layer
    :param num_routing: number of iterations for the routing algorithm
    """

    def __init__(self, num_capsule, dim_vector, num_routing=3,
                 kernel_initializer='glorot_uniform',
                 b_initializer='zeros',
                 **kwargs):
        super(DigiCap, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_vector = dim_vector
        self.num_routing = num_routing
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.b_initializer = initializers.get(b_initializer)

    def build(self, input_shape):
        "The input Tensor should have shape=[None, input_num_capsule, input_dim_vector]"
        assert len(input_shape) >= 3
        self.input_num_capsule = input_shape[1]
        self.input_dim_vector = input_shape[2]

        # Transform matrix W
        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule,
                                 self.input_dim_vector, self.dim_vector],
                                 initializer=self.kernel_initializer,
                                 name='W')
        # Coupling coefficient.
        # The redundant dimensions are just to facilitate subsequent matrix calculation
        self.b = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1],
                                    initializer=self.b_initializer,
                                    name='b',
                                    trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape = (None, input_num_capsule, input_dim_vector)
        # Expand dims to (None, input_num_capsule, 1, 1, input_input_dim_vector)
        inputs_expand = K.expand_dims(K.expand_dims(inputs, 2), 2)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # Now shape = [None, input_num_capsule, num_capsule, 1, input_dim_vector]
        inputs_tiled = K.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])

        # However, we will implement the same code with a faster implementation using tf.sacn	
        # Compute `inputs * W` by scanning inputs_tiled on dimension 0. 
        # inputs_hat.shape = [None, input_num_capsule, num_capsule, 1, dim_vector]
        inputs_hat = tf.scan(lambda ac, x: K.batch_dot(x, self.W, [3, 2]),
                             elems=inputs_tiled,
                             initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_vector]))

        # Routing algorithm
        assert self.num_routing > 0, 'the num_routing should be > 0.'
        for i in range(self.num_routing):
            c = tf.nn.softmax(self.b, dim=2) # dim=2 is the num_capsule dimension
            outputs = squash(K.sum(c * inputs_hat, 1, keepdims=True))

            # last iteration needs not compute b which will not be passed to the graph anymore anyway
            if i != self.num_routing -1:
                self.b += K.sum(inputs_hat * outputs, -1, keepdims=True)
        return K.reshape(outputs, [-1, self.num_capsule, self.dim_vector])


def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_vector: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_vector]
    """
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_vector])(output)
    return layers.Lambda(squash)(outputs)


def CapsNet(input_shape, n_class, num_routing):
    """
    :param input_shape: (None, width, height, channels)
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs (image, label) and
             2 outputs (capsule output and reconstruct image)
    """
    # Image
    x = layers.Input(shape=input_shape)

    # ReLU Conv1
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1,
                    padding='valid', activation='relu', name='conv1')(x)
    
    # PrimaryCapsules: Conv2D layer with `squash` activation
    # reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32,
                    kernel_size=9, strides=2, padding='valid')
    
    # DigiCap: Capsule layer. Routing algorithm works here
    digicaps = DigiCaps(num_capsule=n_class, dim_vector=16,
                    num_routing=num_routing, name='digitcaps')(primarycaps)

    # The length of the capsule's output vector
    out_caps = Length(name='out_caps')(digicaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))

    # The true label is used to extract the corresponding vj
    masked = Mask()([digicaps, y])
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(784, activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=[28,28,1], name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return models.Model([x, y], [out_caps, x_recon])

