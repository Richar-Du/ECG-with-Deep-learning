import tensorflow as tf
def SENet(input_ecg):
    x=tf.keras.layers.Conv1D(filters=64, kernel_size=7, strides=2, padding='same',activation=tf.nn.relu)(input_ecg)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=3)(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=2, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(x)
    x = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=2, padding='same', activation=tf.nn.relu)(x)
    x=SEBlock(x)
    for i in range(4):
        x = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=2, padding='same', activation=tf.nn.relu)(x)
        x = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(x)
        x = tf.keras.layers.Conv1D(filters=512, kernel_size=1, strides=2, padding='same', activation=tf.nn.relu)(x)
        x = SEBlock(x)
    for i in range(6):
        x = tf.keras.layers.Conv1D(filters=256, kernel_size=1, strides=2, padding='same', activation=tf.nn.relu)(x)
        x = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(x)
        x = tf.keras.layers.Conv1D(filters=1024, kernel_size=1, strides=2, padding='same', activation=tf.nn.relu)(x)
        x = SEBlock(x)
    for i in range(3):
        x = tf.keras.layers.Conv1D(filters=512, kernel_size=1, strides=2, padding='same', activation=tf.nn.relu)(x)
        x = tf.keras.layers.Conv1D(filters=512, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(x)
        x = tf.keras.layers.Conv1D(filters=2048, kernel_size=1, strides=2, padding='same', activation=tf.nn.relu)(x)
        x = SEBlock(x)

    # layer2=tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)(SE)
    # BachNorm = tf.keras.layers.BatchNormalization()(layer2)
    # SE = SEBlock(BachNorm)
    # x=tf.keras.layers.MaxPool1D()(x)
    # x=tf.keras.layers.LSTM(10)(x)
    flatten=tf.keras.layers.Flatten()(x)
    dense = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)(flatten)
    dropout = tf.keras.layers.Dropout(rate=0.1)(dense)
    label_ecg = tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)(dropout)
    return label_ecg
# def SENet():
#     return tf.keras.models.Sequential([
#         tf.keras.layers.Conv1D(filters=64, kernel_size=20, strides=3, padding='same',activation=tf.nn.relu),
#         tf.keras.layers.BatchNormalization(),
#         # tf.keras.layers.MaxPool1D(pool_size=2, strides=3),
#         SEBlock(),
#         tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu),
#         tf.keras.layers.BatchNormalization(),
#         SEBlock(),
#         # tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
#         # tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu),
#         # tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu),
#         # tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
#         # tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu),
#         # tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),
#         tf.keras.layers.LSTM(10),
#         tf.keras.layers.Flatten(),
#         # tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
#         tf.keras.layers.Dropout(rate=0.1),
#         tf.keras.layers.Dense(units=20, activation=tf.nn.relu),
#         tf.keras.layers.Dense(units=10, activation=tf.nn.relu),
#         tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)
#     ])
def SEBlock(inputs, reduction=16, if_train=True):
    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    x = tf.keras.layers.Dense(int(x.shape[-1]) // reduction, use_bias=False, activation=tf.keras.activations.relu, trainable=if_train)(x)
    x = tf.keras.layers.Dense(int(inputs.shape[-1]), use_bias=False, activation=tf.keras.activations.hard_sigmoid, trainable=if_train)(x)
    return tf.keras.layers.Multiply()([inputs, x])
