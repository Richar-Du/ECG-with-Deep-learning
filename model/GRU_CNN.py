import tensorflow as tf


# =======================================定义SE-Block=======================================
def SEBlock(inputs,reduction=8,if_train=True):
    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    x = tf.keras.layers.Dense(int(x.shape[-1]//reduction),use_bias=False,activation=tf.keras.activations.relu,trainable=if_train)(x)
    x = tf.keras.layers.Dense(int(inputs.shape[-1]),use_bias=False,activation=tf.keras.activations.hard_sigmoid,trainable=if_train)(x)
    return tf.keras.layers.Multiply()([inputs,x])
# ======================================定义SE-Res-Block===================================
def SERBlock(inputs,reduction=16,if_train=True):
	x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
	x = tf.keras.layers.Dense(int(x.shape[-1]//reduction),use_bias=False,activation=tf.keras.activations.relu,trainable=if_train)(x)
	x = tf.keras.layers.Dense(int(inputs.shape[-1]),use_bias=False,activation=tf.keras.activations.hard_sigmoid,trainable=if_train)(x)
	x = tf.keras.layers.Multiply()([inputs,x])
	return tf.keras.layers.Add()([inputs,x])

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
    tf.keras.layers.Dropout(rate=0.1)(x)
    label_ecg=tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)(x)
    return label_ecg

# ======================================================自定义类的方式==============================
class SE_GRU(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1=tf.keras.layers.Conv1D(filters=128,kernel_size=50,strides=3,padding='same',activation=tf.nn.relu)
        self.BachNorm = tf.keras.layers.BatchNormalization()
        self.MaxPooling1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=3)
        self.conv2 = tf.keras.layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)
        self.MaxPooling2 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        self.conv3 = tf.keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu)
        self.conv4 = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)
        self.MaxPooling3 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        self.conv5 = tf.keras.layers.Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
        self.conv6 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        self.gru1 = tf.keras.layers.GRU(units=60)
        self.flat = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=512, activation=tf.nn.relu)
        self.Dropout = tf.keras.layers.Dropout(rate=0.1)
        self.outputSoftmax=tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)

    def call(self,inputs):
        x=self.conv1(inputs)
        x=self.BachNorm(x)
        x=self.SEBlock(x)
        x=self.MaxPooling1(x)
        x=self.conv2(x)
        x=self.BachNorm(x)
        x=self.SEBlock(x)
        x=self.MaxPooling2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.SEBlock(x)
        x=self.MaxPooling3(x)
        x=self.conv5(x)
        x=self.conv6(x)
        x=self.SEBlock(x)
        x=self.gru1(x)
        x=self.flat(x)
        x=self.dense(x)
        x=self.Dropout(x)
        output=self.outputSoftmax(x)
        return output
# ======================================================sequential的方式==============================================
def Seq_SE_GRU(ecg_input):
	x = tf.keras.layers.Conv1D(filters=256, kernel_size=50, strides=3, padding='same',activation=tf.nn.relu)(ecg_input)
	x = SEBlock(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.MaxPool1D(pool_size=2, strides=3)(x)
	x = tf.keras.layers.Conv1D(filters=128, kernel_size=7, strides=1, padding='same', activation=tf.nn.relu)(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)(x)
	x = tf.keras.layers.Conv1D(filters=64,kernel_size=5,strides=1,padding='same',activation=tf.nn.relu)(x)
	x = SEBlock(x)
	x = tf.keras.layers.BatchNormalization()(x)
	x = tf.keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
	x = tf.keras.layers.Conv1D(filters=64, kernel_size=10, strides=1, padding='same', activation=tf.nn.relu)(x)
	x = tf.keras.layers.Conv1D(filters=128, kernel_size=5, strides=2, padding='same', activation=tf.nn.relu)(x)
	x = tf.keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
	x = tf.keras.layers.Conv1D(filters=256,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)(x)
	x = tf.keras.layers.Conv1D(filters=128,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu)(x)
	x = SEBlock(x)
	x = tf.keras.layers.MaxPool1D(pool_size=2,strides=2)(x)
	x = tf.keras.layers.GRU(units=70)(x)
	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.Dense(units=45, activation=tf.nn.relu)(x)
	x = tf.keras.layers.Dropout(rate=0.3)(x)
	output = tf.keras.layers.Dense(units=7, activation=tf.nn.softmax)(x)
	
	return output

def SeqGRU():
	return tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=3, kernel_size=20, strides=1, activation=tf.nn.relu),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=6, kernel_size=10, strides=1, activation=tf.nn.relu),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.Conv1D(filters=6, kernel_size=5, strides=1, activation=tf.nn.relu),
        #tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool1D(pool_size=2, strides=2),
        tf.keras.layers.GRU(units=24),
        tf.keras.layers.Dense(24,activation=tf.nn.relu),
        tf.keras.layers.Dense(14,activation=tf.nn.relu),
        #tf.keras.layers.Dropout(rate=0.1),
        tf.keras.layers.Dense(7,activation=tf.nn.softmax)
        ])
