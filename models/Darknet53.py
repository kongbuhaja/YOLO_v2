import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Softmax, Reshape, Layer, Add
from tensorflow.keras.initializers import GlorotUnifrom as glorot
from tensorflow.keras.models import Model

class Residual_block(Layer):
    def __init__(self, units, name="", kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.name = name
        self.kernel_initializer = kernel_initializer

    def call(self, input):
        x = Conv(self.units/2, name=self.name[0], kernel_initializer=self.kernel_initializer)(x)
        x = Conv(self.units, (1,1), name=self.name[1], kernel_initializer=self.kernel_initializer)(x)
        return Add()([input, x])

class Conv(Layer):
    def __init__(self, units, kernel_size=(3,3), strides=(1,1), name="", kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.name = name
        self.kernel_initializer = kernel_initializer

    def call(self, input):
        x = Conv2D(self.unit, self.kernel_size, padding="same", strides=self.strides, name="conv_"+self.name, kernel_initializer=self.kernel_initializer)(input)
        x = BatchNormalization(name="norm_"+self.name)(x)
        x = LeakyReLU(alpha=0.1)(x)
        return x

def get_model(labels):
    input = Input(shape=(None, None, 3))
    x = Conv(32, name="1")(x)

    x = Conv(64, strides=(2,2), name="2")(x)
    x = Residual_block(64, name=["3","4"])(x)

    x = Conv(128, strides=(2,2), name="5")(x)
    for i in range(6, 10, 2):
        x = Residual_block(128, name=[str(i), str(i+1)])(x)

    x = Conv(256, strides=(2,2), name="10")(x)
    for i in range(11, 27, 2):
        x = Residual_block(256, name=[str(i), str(i+1)])(x)

    x = Conv(512, strides=(2,2), name="27")(x)
    for i in range(28, 44, 2):
        x = Residual_block(512, name=[str(i), str(i+1)])(x)

    x = Conv(1024, strides=(2,2), name="44")(x)
    for i in range(45, 53, 2):
        x = Residual_block(1024, name=[str(i), str(i+1)])(x)