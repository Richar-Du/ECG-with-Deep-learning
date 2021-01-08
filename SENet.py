import os
import seaborn as sns
import collections
import numpy as np
import scipy.io as scio

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, \
    BatchNormalization, Multiply, Layer, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.activations as activations
import tensorflow.keras.layers as layers

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
#=================================================== 读取数据================================================
np.random.seed(7)
choose_index=np.random.randint(1,100,100)
print("load data...")
X_list=['/home/group5/new_new_data/X_eu_MLIII.csv','/home/group5/new_new_data/X_ST_ECG.csv','/home/group5/new_new_data/X_SU30_ECG.csv','/home/group5/new_new_data/X_SU31_ECG.csv','/home/group5/new_new_data/X_SU32_ECG.csv','/home/group5/new_new_data/X_SU34_ECG.csv','/home/group5/new_new_data/X_SU35_ECG.csv','/home/group5/new_new_data/X_SU36_ECG.csv','/home/group5/new_new_data/X_SU41_ECG.csv','/home/group5/new_new_data/X_SU45_ECG.csv','/home/group5/new_new_data/X_SU46_ECG.csv','/home/group5/new_new_data/X_SU49_ECG.csv','/home/group5/new_new_data/X_SU51_ECG.csv','/home/group5/new_new_data/X_SU52_ECG.csv']
Y_list=['/home/group5/new_new_data/Y_eu_MLIII.csv','/home/group5/new_new_data/Y_ST_ECG.csv','/home/group5/new_new_data/Y_SU30_ECG.csv','/home/group5/new_new_data/Y_SU31_ECG.csv','/home/group5/new_new_data/Y_SU32_ECG.csv','/home/group5/new_new_data/Y_SU34_ECG.csv','/home/group5/new_new_data/Y_SU35_ECG.csv','/home/group5/new_new_data/Y_SU36_ECG.csv','/home/group5/new_new_data/Y_SU41_ECG.csv','/home/group5/new_new_data/Y_SU45_ECG.csv','/home/group5/new_new_data/Y_SU46_ECG.csv','/home/group5/new_new_data/Y_SU49_ECG.csv','/home/group5/new_new_data/Y_SU51_ECG.csv','/home/group5/new_new_data/Y_SU52_ECG.csv']
X=np.loadtxt('data/X_MIT.csv',delimiter=',',skiprows=1).astype('float32')#[choose_index]
Y=np.loadtxt('data/Y_MIT.csv',dtype="str",delimiter=',',skiprows=1)#[choose_index]
#合并数据集
# print("begin concatenating...")
# for database in X_list:
#     X=np.concatenate((X,(np.loadtxt(database,dtype="str",delimiter=',',skiprows=1).astype(np.float))))
# for database in Y_list:
#     Y=np.concatenate((Y,(np.loadtxt(database,dtype="str",delimiter=',',skiprows=1))))

AAMI=['L','R','V','A','|','B']
# N:Normal
# L:Left bundle branch block beat
# R:Right bundle branch block beat
# V:Premature ventricular contraction
# A:Atrial premature contraction
# |:Isolated QRS-like artifact
# B:Left or right bundle branch block
delete_list=[]
for i in range(len(Y)):
    if Y[i] not in AAMI:            # 删除不在AAMI中标签的数据
        delete_list.append(i)
X=np.delete(X,delete_list,0)
Y=np.delete(Y,delete_list,0)
#保存用于训练的数据
# savedX=pd.DataFrame(X)
# savedY=pd.DataFrame(Y)
# savedX.to_csv('/data/Concatenated_X.csv', index=False)
# savedY.to_csv('/data/Concatenated_Y.csv', index=False)
#数据标准化：
print("begin standard scaler...")
ss = StandardScaler()
std_data = ss.fit_transform(X)
X=np.expand_dims(X,axis=2)


# 把标签编码
le=preprocessing.LabelEncoder()
le=le.fit(AAMI)
Y=le.transform(Y)
# Y=np.expand_dims(Y,axis=3)
print("the label before encoding:",le.inverse_transform([0,1,2,3,4,5,6]))
# 分层抽样
print("begin StratifiedShuffleSplit...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, train_size=0.9,random_state=0)   # n_split=1就只有二八分，如果需要交叉验证，把训练和测试的代码放到for循环里面就可以
sss.get_n_splits(X, Y)
for train_index, test_index in sss.split(X, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    y_train=np.eye(7)[y_train]
def get_batch(X,y,batch_size):
    index = np.random.randint(0, len(y), batch_size)
    return X[index, :], y[index]

## 2.2 卷积降噪自编码器训练
# 为原数据加入噪声
## 噪声影响程度
factor = 0.05
noise = np.random.randn(len(X_train), len(X_train[0])) * factor
X_train += noise

# 定义SEBlock
## reduction 用来降维
def SEBlock(inputs, reduction=4, if_train=True):
    x = GlobalAveragePooling1D()(inputs)
    x = Dense(int(x.shape[-1]) // reduction, use_bias=False, activation=activations.relu, trainable=if_train)(x)
    x = Dense(int(inputs.shape[-1]), use_bias=False, activation=activations.hard_sigmoid, trainable=if_train)(x)
    return Multiply()([inputs, x])

# 定义卷积降噪自编码器
def conv_autoencoder(input_ecg):
    x = layers.Conv1D(filters=16, kernel_size=3, padding='same', name='encoder_1')(input_ecg)
    x = BatchNormalization(trainable=False)(x)
    x = Activation('relu')(x)
    x = layers.MaxPool1D(pool_size=2, padding='same', name='encoder_2')(x)
    encoded = SEBlock(x)

    x = layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same', name='decoder_1')(encoded)
    x = layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same', name='decoder_3')(x)
    x = layers.UpSampling1D(size=2, name='decoder_4')(x)
    decoded = layers.Conv1D(filters=1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_5')(x)

    autoencoder = Model(inputs=input_ecg, outputs=decoded)

    return autoencoder


input_ecg = Input(shape=(3600, 1))
autoencoder = conv_autoencoder(input_ecg=input_ecg)
autoencoder.summary()
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# 训练降噪自编码器
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


# 打开一个终端并启动TensorBoard，终端中输入 tensorboard --logdir=/autoencoder
autoencoder.fit(X_train, X_train, epochs=2, batch_size=32,  shuffle=True)


## 2.3 混合模型定义


#### 加载降噪自编码器


def vgg19_heart(input_data, classes=7):
    x = layers.Conv1D(filters=128, kernel_size=3, padding='same', name='encoder_1')(input_ecg)
    x = BatchNormalization(trainable=False)(x)
    x = Activation('relu')(x)
    x = layers.MaxPool1D(pool_size=2, padding='same', name='encoder_2')(x)
    x = SEBlock(x)
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', name='encoder_3')(x)
    x = BatchNormalization(trainable=False)(x)
    x = Activation('relu')(x)
    x = layers.MaxPool1D(pool_size=2, padding='same', name='encoder_4')(x)
    x = SEBlock(x)

    # Block2
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', name='block2_conv1', activation='relu')(x)
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', name='block2_conv2', activation='relu')(x)
    x = SEBlock(x)
    x = layers.MaxPool1D(pool_size=2, strides=2, name='block2_pool')(x)

    x = layers.Conv1D(filters=128, kernel_size=3, padding='same', name='block3_conv1', activation='relu')(x)
    x = layers.MaxPool1D(pool_size=2, strides=2, name='block3_pool_1')(x)
    x = layers.Conv1D(filters=128, kernel_size=3, padding='same', name='block3_conv2', activation='relu')(x)
    x = layers.MaxPool1D(pool_size=2, strides=2, name='block3_pool_2')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(input_data, x, name='VGG19-Heart')

    return model

model = vgg19_heart(input_data=input_ecg, classes=len(AAMI))
print(model.summary())
for i in range(1,9):
    model.layers[i].set_weights(autoencoder.layers[i].get_weights())


# setting optimizers & compile
optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model_all.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# expand X_train dims
# Y : int -> binary (one-hot)
# y_train = to_categorical(y_train,num_classes = ClassesNum)
# y_test = to_categorical(y_test,num_classes = ClassesNum)

# display(np.shape(X_train))


# 加载模型
model = load_model('model.h5')


BATCH_SIZE = 16
EPOCHS = 100
# history = model_all.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = EPOCHS)
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2)


model.save('./model.h5')

# %%

# val_loss_acc = model_all.evaluate(X_test, y_test, batch_size=100)
val_loss_acc = model.evaluate(X_test, y_test, batch_size=16)
print("loss of val : ", val_loss_acc[0])
print("acc of val : ", val_loss_acc[1])




plt.rcParams["figure.figsize"] = (15, 5)
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.color'] = 'b'
plt.rcParams['axes.grid'] = True

predictions = model.predict(X_test)
print(predictions)

y_pre = np.zeros((int(predictions.size / predictions[0].size), predictions[0].size))
for i in range(len(predictions)):
    y_pre[i][np.where(predictions[i] == max(predictions[i]))[0][0]] = 1
print(y_pre[0])



def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
    predictions = model.predict(x_val)
    truelabel = y_val.argmax(axis=-1)  # 将one-hot转化为label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    plt.figure()
    plot_confusion_matrix(conf_mat, range(np.max(truelabel) + 1))


# %%

pre_label = y_pre.argmax(axis=-1)
test_label = y_test.argmax(axis=-1)
conf_mat = confusion_matrix(y_true=test_label, y_pred=pre_label)
sns.heatmap(conf_mat, annot=True)
