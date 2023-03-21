from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, LeakyReLU, LSTM, Bidirectional, Concatenate, Conv2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.math import reduce_sum


def CNN(shape_input):
    inputs = Input(shape=shape_input, name='Input_1')
    t = Conv2D(filters=64, kernel_size=3, padding="same", activation='relu', kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01), name='Conv_1')(inputs)
    t = Conv2D(filters=64, kernel_size=3, padding="same", activation='relu', kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01), name='Conv_2')(t)
    t = MaxPooling2D(pool_size=2, strides=2, name='MaxPooling_1')(t)
    t = Conv2D(filters=128, kernel_size=3, padding="same", activation='relu', kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01), name='Conv_3')(t)
    t = Conv2D(filters=128, kernel_size=3, padding="same", activation='relu', kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01), name='Conv_4')(t)
    t = MaxPooling2D(pool_size=2, strides=2, name='MaxPooling_2')(t)
    t = Conv2D(filters=256, kernel_size=3, padding="same", activation='relu', kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01), name='Conv_5')(t)
    t = Conv2D(filters=256, kernel_size=3, padding="same", activation='relu', kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01), name='Conv_6')(t)
    t = Conv2D(filters=256, kernel_size=3, padding="same", activation='relu', kernel_regularizer=l2(0.01),
               bias_regularizer=l2(0.01), name='Conv_7')(t)
    t = MaxPooling2D(pool_size=2, strides=2, name='MaxPooling_3')(t)
    t = Flatten(name='Flatten')(t)
    t = Dense(256, activation='relu', name='Danse_1')(t)
    t = Dense(1, name='Danse_2')(t)
    return Model(inputs, t)