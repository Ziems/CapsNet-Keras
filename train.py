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

