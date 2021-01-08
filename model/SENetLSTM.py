import tensorflow as tf
def SENet_LSTM(ecg_input):
    x=tf.keras.layers.Conv1D(filters=128, kernel_size=20, strides=3, padding='same',activation=tf.nn.relu)(ecg_input)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPool1D(pool_size=2, strides=3)(x)
    x=tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)(x)
    x=SEBlock(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    x=tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = SEBlock(x)
    # tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu),
    x=tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
    # tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu),
    # tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),
    x=tf.keras.layers.LSTM(10)(x)
    x=tf.keras.layers.Flatten()(x)
    # tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
    x=tf.keras.layers.Dropout(rate=0.1)(x)
    x=tf.keras.layers.Dense(units=20, activation=tf.nn.relu)(x)
    x=tf.keras.layers.Dense(units=10, activation=tf.nn.relu)(x)
    output=tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)(x)
    return output

def SEBlock(inputs, reduction=16, if_train=True):
    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    x = tf.keras.layers.Dense(int(x.shape[-1]) // reduction, use_bias=False, activation=tf.keras.activations.relu, trainable=if_train)(x)
    x = tf.keras.layers.Dense(int(inputs.shape[-1]), use_bias=False, activation=tf.keras.activations.hard_sigmoid, trainable=if_train)(x)
    return tf.keras.layers.Multiply()([inputs, x])