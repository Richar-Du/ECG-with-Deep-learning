# -*- coding: UTF-8 -*-
#====================================================== 导入需要的包==================================
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd
import os
import platform
import datetime
import tensorflow as tf

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn import preprocessing
# 调用自己写的模型架构
from model.CNN import CNN1d
from model.CNN import CNN
from model.CNN import SeqCNN
from model.CNNLSTM import CNN_LSTM
from model.SE_Net import SENet
from model.SENetLSTM import SENet_LSTM
#=================================================== 读取数据================================================
np.random.seed(7)
choose_index=np.random.randint(1,100,100)
print("load data...")
X_list=['/home/group5/new_new_data/X_eu_MLIII.csv','/home/group5/new_new_data/X_ST_ECG.csv','/home/group5/new_new_data/X_SU30_ECG.csv','/home/group5/new_new_data/X_SU31_ECG.csv','/home/group5/new_new_data/X_SU32_ECG.csv','/home/group5/new_new_data/X_SU34_ECG.csv','/home/group5/new_new_data/X_SU35_ECG.csv','/home/group5/new_new_data/X_SU36_ECG.csv','/home/group5/new_new_data/X_SU41_ECG.csv','/home/group5/new_new_data/X_SU45_ECG.csv','/home/group5/new_new_data/X_SU46_ECG.csv','/home/group5/new_new_data/X_SU49_ECG.csv','/home/group5/new_new_data/X_SU51_ECG.csv','/home/group5/new_new_data/X_SU52_ECG.csv']
Y_list=['/home/group5/new_new_data/Y_eu_MLIII.csv','/home/group5/new_new_data/Y_ST_ECG.csv','/home/group5/new_new_data/Y_SU30_ECG.csv','/home/group5/new_new_data/Y_SU31_ECG.csv','/home/group5/new_new_data/Y_SU32_ECG.csv','/home/group5/new_new_data/Y_SU34_ECG.csv','/home/group5/new_new_data/Y_SU35_ECG.csv','/home/group5/new_new_data/Y_SU36_ECG.csv','/home/group5/new_new_data/Y_SU41_ECG.csv','/home/group5/new_new_data/Y_SU45_ECG.csv','/home/group5/new_new_data/Y_SU46_ECG.csv','/home/group5/new_new_data/Y_SU49_ECG.csv','/home/group5/new_new_data/Y_SU51_ECG.csv','/home/group5/new_new_data/Y_SU52_ECG.csv']
X=np.loadtxt('data/X_MIT.csv',delimiter=',',skiprows=1).astype('float32')#[choose_index]
Y=np.loadtxt('data/Y_MIT.csv',dtype="str",delimiter=',',skiprows=1)#[choose_index]
#合并数据集
print("begin concatenating...")
for database in X_list:
    X=np.concatenate((X,(np.loadtxt(database,dtype="str",delimiter=',',skiprows=1).astype(np.float))))
for database in Y_list:
    Y=np.concatenate((Y,(np.loadtxt(database,dtype="str",delimiter=',',skiprows=1))))

AAMI=['N','L','R','V','A','|','B']
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
#===================================================模型训练==================================================
# 定义超参数
num_epochs = 60
batch_size = 32
learning_rate = 0.001
#=======================================一维 CNN模型====================================================
# =======================================直接调用的方式===================================
input_ecg=tf.keras.layers.Input(shape=(3600,1))
label_ecg=SENet_LSTM(input_ecg)
print("begin 1dCNN")
model=Model(input_ecg,label_ecg)
print('model summary:',model.summary())
# 设置优化器
optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999,name='Adam')
loss=tf.keras.losses.categorical_crossentropy
metrics=['accuracy']
# 初始化tensorboard
# 注意windows系统和linux系统下文件路径的问题
if platform.system()=='Windows':
    log_dir="tensorboard\\fit\\CNN" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
elif platform.system()=='Linux':
    log_dir="tensorboard/fit/CNN" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer=tf.summary.create_file_writer(log_dir)   # 实例化一个记录器
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# 配置训练过程
print('begin compile CNN')
model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
# 训练模型
print("begin fit CNN")
savedModule=model.fit(X_train,y_train,epochs=num_epochs,batch_size=batch_size,validation_split=0.2,callbacks=[tensorboard_callback])
plt.plot(savedModule.epoch,savedModule.history['loss'],color='blue',label='train loss')
plt.plot(savedModule.epoch,savedModule.history['val_loss'],color="red",label='test loss')
plt.legend()
plt.show()
# 评估训练效果
print(model.evaluate(X_test,y_test))
predict=model.predict(X_test)       # 输出的不是一个类别，而是样本属于每一个类别的概率
predict=[np.argmax(predict[i]) for i in range(len(predict))]
print('confusion matrix:',tf.math.confusion_matrix(y_test,predict))
# 保存模型
if platform.system()=='Windows':
    tf.saved_model.save(model, "save\\CNN")
elif platform.system()=='Linux':
    tf.saved_model.save(model,"save/CNN")
# ======================================================自定义类的方式==============================
# model=CNN()
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999,name='Adam')
# # 初始化tensorboard
# # 注意windows系统和linux系统下文件路径的问题
# if platform.system()=='Windows':
#     log_dir="tensorboard\\fit\\CNN" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# elif platform.system()=='Linux':
#     log_dir="tensorboard/fit/CNN" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# summary_writer=tf.summary.create_file_writer(log_dir)   # 实例化一个记录器
# num_batches = int(len(y_train) // batch_size * num_epochs)
# for batch_index in range(num_batches):
#     X,y=get_batch(X_train,y_train,batch_size)
#     with tf.GradientTape() as tape:
#         y_pred=model(X)
#         loss=tf.keras.losses.categorical_crossentropy(y_true=y,y_pred=y_pred)
#         loss=tf.reduce_mean(loss)
#         print("batch %d: loss %f" % (batch_index, loss.numpy()))
#     grads = tape.gradient(loss, model.variables)
#     optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
#     with summary_writer.as_default():  # 希望使用的记录器
#         tf.summary.scalar("loss", loss, step=batch_index)
        # tf.summary.scalar("MyScalar", my_scalar, step=batch_index)  # 还可以添加其他自定义的变量
# # 保存模型
# if platform.system()=='Windows':
#     tf.saved_model.save(model, "save\\CNN")
# elif platform.system()=='Linux':
#     tf.saved_model.save(model,"save/CNN")
# ======================================================sequential的方式==============================
# model=SENet()
# print("begin optimizer...")
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,beta_1=0.9,beta_2=0.999,name='Adam')
# print("begin reduce_lr...")
# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                               patience=5, min_lr=0.001)
# print("begin checkpoint...")
# checkpoint = tf.train.Checkpoint(SeqCNN_model=model,SeqCNN_optimizer=optimizer)
# if platform.system()=='Windows':
#     ModelSavedPath="save\\CNN"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# elif platform.system()=='Linux':
#     ModelSavedPath="save/CNN"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# checkpoint=tf.keras.callbacks.ModelCheckpoint(
#     ModelSavedPath, monitor='val_loss', verbose=0, save_best_only=True,
#     save_weights_only=False, mode='auto', save_freq='epoch', options=None, )
# print("begin compiling...")
# model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
# if platform.system()=='Windows':
#     log_dir="tensorboard\\fit\\CNN"  + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# elif platform.system()=='Linux':
#     log_dir="tensorboard/fit/CNN"  + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# # summary_writer = tf.summary.create_file_writer(log_dir)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# print("begin fitting...")
# history=model.fit(x=X_train,
#           y=y_train,
#           epochs=num_epochs,
#           batch_size=batch_size,
#           validation_split=0.2,
#           callbacks=[tensorboard_callback,reduce_lr])
# # with summary_writer.as_default():  # 指定记录器
# #     tf.summary.scalar("loss", history.history['loss'], step=history.epoch)  # 将当前损失函数的值写入记录器
# print("history loss=",history.history['loss'])
# # 评估训练效果
# print(model.evaluate(X_test,y_test))
# predict=model.predict(X_test)       # 输出的不是一个类别，而是样本属于每一个类别的概率
# predict=[np.argmax(predict[i]) for i in range(len(predict))]
# print('confusion matrix:',tf.math.confusion_matrix(y_test,predict))
# plt.subplot(1,2,1) #要生成两行两列，这是第一个图plt.subplot('行','列','编号')
# plt.plot(history.epoch,history.history['loss'],color='b',label='train loss')
# plt.plot(history.epoch,history.history['val_loss'],color="red",label='validation loss')
# plt.legend()
# plt.subplot(1,2,2)
# plt.plot(history.epoch,history.history['accuracy'],color='b',label='train acc')
# plt.plot(history.epoch,history.history['val_accuracy'],color="red",label='validation acc')
# plt.legend()
# plt.title("loss and accuracy")
# if platform.system()=='Windows':
#     plt.savefig("figure\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'CNN1d.png')
# elif platform.system()=='Linux':
#     plt.savefig("figure/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'CNN1d.png')
# # plt.show()
# ==========================================================保存模型并转化成tensoflow lite格式的文件==========================================================
# 使用checkpoint保存模型参数
print("begin saving checkpoint...")
checkpoint.save("checkpoint/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"SeqCNN.ckpt")
# 使用 SavedModel 完整保存模型
if platform.system()=='Windows':
    tf.saved_model.save(model, "save\\CNN"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
elif platform.system()=='Linux':
    tf.saved_model.save(model,"save/CNN"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# 当需要载入模型时：
# model = tf.saved_model.load("保存的目标文件夹名称")
# 转化模型格式
if platform.system()=='Windows':
    tflite_path="tflite\\model"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".tflite"
elif platform.system()=='Linux':
    tflite_path = "tflite/model"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(ModelSavedPath) # path to the SavedModel directory
tflite_model = converter.convert()
# Save the model.
with open(tflite_path, 'wb') as f:
  f.write(tflite_model)
f.close()


