import wfdb     #导入wfdb包读取数据文件
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy
from scipy import signal

type=[]
rootdir = 'european-st-t-database-1.0.0'
# rootdir = 'mit-bih-st-change-database-1.0.0'     #欧盟st-t数据库
# rootdir = 'sudden-cardiac-death-holter-database-1.0.0'        #心脏性猝死数据库

files = os.listdir(rootdir) #列出文件夹下所有
name_list=[]            # name_list=[100,101,...234]
MLIII=[]                 # 用MLIII型导联采集的人（根据选择的不同导联方式会有变换）
type={}                 # 标记及其数量
for file in files:
    if file[0:5] in name_list:     # 选取文件的前五个字符（可以根据数据文件的命名特征进行修改）
        continue
    else:
        name_list.append(file[0:5])
for name in name_list:      # 遍历每一个人
    if name[1] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:       # 判断——跳过无用的文件
        continue
    record = wfdb.rdrecord(rootdir+'/'+name)  # 读取一条记录（100），不用加扩展名
    if 'MLIII' in record.sig_name:       # 这里我们记录MLIII导联的数据（也可以记录其他的，根据数据库的不同选择数据量多的一类导联方式即可）
        MLIII.append(name)               # 记录下这个人
    annotation = wfdb.rdann(rootdir+'/'+name, 'atr')  # 读取一条记录的atr文件，扩展名atr
    for symbol in annotation.symbol:            # 记录下这个人所有的标记类型
        if symbol in list(type.keys()):
            type[symbol]+=1
        else:
            type[symbol]=1
    print('sympbol_name',type)
sorted(type.items(),key=lambda d:d[1],reverse=True)

f=250       # 数据库的原始采样频率
segmented_len=10        # 将数据片段裁剪为10s
label_count=0
count=0
abnormal=0

segmented_data = []             # 最后数据集中的X
segmented_label = []            # 最后数据集中的Y
print('begin!')

for person in MLIII:        # 读取导联方式为MLIII的数据
    k = 0
    whole_signal=wfdb.rdrecord(rootdir + '/' + person).p_signal.transpose()     # 这个人的一整条数据
    while (k+1)*f*segmented_len<=len(whole_signal[0]):    # 只要不到最后一组数据点
        count+=1
        record = wfdb.rdrecord(rootdir + '/' + person, sampfrom=k * f * segmented_len,sampto=(k + 1) * f * segmented_len)  # 读取一条记录（100），不用加扩展名
        annotation = wfdb.rdann(rootdir + '/' + person, 'atr', sampfrom=k * f * segmented_len,sampto=(k + 1) * f * segmented_len)  # 读取一条记录的atr文件，扩展名atr
        lead_index = record.sig_name.index('MLIII')  # 找到MLII导联对应的索引
        signal = record.p_signal.transpose()  # 两个导联，转置之后方便画图
        label=[]           # 这一段数据对应的label，最后从这里面选择最终的label
        # segmented_data.append(signal[lead_index])   # 只记录MLII导联的数据段
        symbols=annotation.symbol

        re_signal = scipy.signal.resample(signal[lead_index], 3600)  # 采样
        re_signal_3 = np.round(re_signal, 3)
        print('resignal', re_signal_3)
        segmented_data.append(re_signal_3)

        # segmented_data.append(re_signal)
        print('symbols', symbols, len(symbols))


        # if '+' in symbols:  # 删去+
        #     symbols.remove('+')
        if len(symbols) == 0:
            segmented_label.append('Q')
        elif symbols.count('N') / len(symbols) == 1 or symbols.count('N') + symbols.count('/') == len(symbols):  # 如果全是'N'或'/'和'N'的组合，就标记为N
            segmented_label.append('N')
        else:
            for i in symbols:
                if i != 'N':
                    label.append(i)
            segmented_label.append(label[0])

        # print(label)
        k+=1
print('begin to save dataset!')


segmented_data=pd.DataFrame(segmented_data)
segmented_label=pd.DataFrame(segmented_label)
segmented_data.to_csv('30X_eu_MLIII.csv', index=False)
segmented_label.to_csv('30Y_eu_MLIII.csv', index=False)

print('Finished!')