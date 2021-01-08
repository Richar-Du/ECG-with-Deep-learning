import tensorflow as tf
# =======================================直接调用的方式=======================================
def CNN1d(input_ecg):
    layer1=tf.keras.layers.Conv1D(filters=128,kernel_size=50,strides=3,padding='same',activation=tf.nn.relu)(input_ecg)
    BachNorm=tf.keras.layers.BatchNormalization()(layer1)
    MaxPooling1=tf.keras.layers.MaxPool1D(pool_size=2,strides=3)(BachNorm)
    layer2=tf.keras.layers.Conv1D(filters=32,kernel_size=7,strides=1,padding='same',activation=tf.nn.relu)(MaxPooling1)
    BachNorm = tf.keras.layers.BatchNormalization()(layer2)
    MaxPooling2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(BachNorm)
    layer3=tf.keras.layers.Conv1D(filters=32,kernel_size=10,strides=1,padding='same',activation=tf.nn.relu)(MaxPooling2)
    layer4=tf.keras.layers.Conv1D(filters=128,kernel_size=5,strides=2,padding='same',activation=tf.nn.relu)(layer3)
    MaxPooling3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(layer4)
    layer5 = tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)(MaxPooling3)
    layer6 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(layer5)
    flat=tf.keras.layers.Flatten()(layer6)
    x = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)(flat)
    x=tf.keras.layers.Dropout(rate=0.1)(x)
    label_ecg=tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)(x)
    return label_ecg

# ======================================================自定义类的方式==============================
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layer1=tf.keras.layers.Conv1D(filters=128,kernel_size=50,strides=3,padding='same',activation=tf.nn.relu)
        self.BachNorm1 = tf.keras.layers.BatchNormalization()
        self.MaxPooling1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=3)
        self.layer2 = tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)
        self.BachNorm2 = tf.keras.layers.BatchNormalization()
        self.MaxPooling2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        self.layer3 = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu)
        self.layer4 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)
        self.MaxPooling3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        self.layer5 = tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
        self.layer6 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.flat = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.Dropout = tf.keras.layers.Dropout(rate=0.1)
        self.outputSoftmax=tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)
    def call(self,inputs):
        x=self.layer1(inputs)
        x=self.BachNorm1(x)
        x=self.MaxPooling1(x)
        x=self.layer2(x)
        x=self.BachNorm2(x)
        x=self.MaxPooling2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.MaxPooling3(x)
        x=self.layer5(x)
        x=self.layer6(x)
        x=self.flat(x)
        x=self.dense(x)
        x=self.Dropout(x)
        output=self.outputSoftmax(x)
        return output
# ======================================================sequential的方式==============================================
def SeqCNN():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=128, kernel_size=50, strides=3, padding='same',activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=3),
        tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)
    ])