import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Softmax, Reshape
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.models import Model

def get_model(labels):
    input = Input(shape=(None, None, 3))
    x = Conv2D(32, (3,3), padding='same', name='conv_1', kernel_initializer=glorot)(input)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)

    x = Conv2D(64, (3,3), padding='same', name='conv_2', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)

    x = Conv2D(128, (3,3), padding='same', name='conv_3', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    x = Conv2D(64, (1,1), padding='same', name='conv_4', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(128, (3,3), padding='same', name='conv_5', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D((2,2), strides=2)(x)

    x = Conv2D(256, (3,3), padding='same', name='conv_6', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(128, (3,3), padding='same', name='conv_7', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, (3,3), padding='same', name='conv_8', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)

    x = Conv2D(512, (3,3), padding='same', name='conv_9', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)
    
    x = Conv2D(256, (1,1), padding='same', name='conv_10', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(512, (3,3), padding='same', name='conv_11', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(256, (1,1), padding='same', name='conv_12', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(512, (3,3), padding='same', name='conv_13', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)

    passthrough = x

    x = Conv2D(1024, (3,3), padding='same', name='conv_14', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(512, (3,3), padding='same', name='conv_15', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(1024, (3,3), padding='same', name='conv_16', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(512, (3,3), padding='same', name='conv_17', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(1024, (3,3), padding='same', name='conv_18', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # x = Conv2D(1000, (1,1), padding='same', name='conv_19')(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Softmax()(x)

    x = Conv2D(1024, (3,3), padding='same', name='conv_19', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(1024, (3,3), padding='same', name='conv_20', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(1024, (3,3), padding='same', name='conv_21', kernel_initializer=glorot)(x)
    x = BatchNormalization(name='norm_21')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x_concat = tf.concat([x, passthrough], -1)

    output = Conv2D(5*(5+labels), (1,1), padding='same', name='conv_22', kernel_initializer=glorot)(x_concat)
    output = Reshape((13,13,5,5+labels))(output)

    model = Model(inputs=input, outputs=output)
    return model

def init_model(model):
    model(tf.random.uniform((1, 416, 416, 3)))