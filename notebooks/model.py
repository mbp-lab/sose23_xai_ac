
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.legacy import Adam


def cnn_model(input_shape, num_classes):
    model= tf.keras.models.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(64,(3,3), padding='same', activation='relu' ))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128,(5,5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512,(3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.01), name='final_conv_layer'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten()) 
    model.add(Dense(256,activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(512,activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer = Adam(lr=0.0001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
      )
    
    return model


class BN_Inference(tf.keras.layers.Layer):
    def __init__(self, gamma, beta, epsilon, moving_mean, moving_var):
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.moving_mean = moving_mean
        self.moving_var = moving_var
       

    def call(self, inputs):
        return  self.gamma * (inputs - self.moving_mean) / tf.math.sqrt(self.moving_var+self.epsilon) + self.beta
    
def create_bn_replacment(layers, idx):
    gamma = layers[idx].gamma
    beta = layers[idx].beta
    eps = layers[idx].epsilon
    mean = layers[idx].moving_mean
    mean = layers[idx].moving_mean
    var = layers[idx].moving_variance
 
    layers[idx] = BN_Inference(gamma=gamma, beta=beta, epsilon=eps, moving_mean=mean, moving_var=var)
    return layers
