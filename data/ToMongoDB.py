from pymongo import MongoClient
import csv

# 创建连接MongoDB数据库函数
def connection():
    # 1:连接本地MongoDB数据库服务
    conn=MongoClient("localhost")
    # 2:连接本地数据库(guazidata)。没有时会自动创建
    db=conn.ECGdb
    # 3:创建集合
    set1=db.ECG
    # 4:看情况是否选择清空(两种清空方式，第一种不行的情况下，选择第二种)
    #第一种直接remove
    # set1.remove(None)
    #第二种remove不好用的时候
    # set1.delete_many({})
    return set1
def insertToMongoDB(set1):
    with open(' ', 'r', encoding='utf-8')as csvfile:
        # 调用csv中的DictReader函数直接获取数据为字典形式
        reader = csv.DictReader(csvfile)
        array_y=[]
        for label in reader:
            for i in label:
                array_y.append(label[i])

    with open(' ', 'r', encoding='utf-8')as csvfile:
        # 调用csv中的DictReader函数直接获取数据为字典形式
        reader = csv.DictReader(csvfile)
        # 创建一个counts计数一下 看自己一共添加了了多少条数据
        counts = 0
        for each in reader:
            array_x = []
            for k in each:
                # print(each[k])
                array_x.append(float(each[k]))
            data = {
                'fs':360,     #采样频率
                'from':'MIT-BIH arrhythmia Database',    #数据的出处
                'lead':'MLII',      #导联方式
                'p_signal':array_x,  #数据点
                'label':array_y[counts],   #标签值
                'resample':False    #是否经过重采样
            }
            print(counts)
            print(array_x)
            print(array_y[counts])

            counts += 1
            set1.insert(data)
            print('成功添加了' + str(counts) + '条数据 ')
# 创建主函数
def main():
    set1=connection()
    insertToMongoDB(set1)
# 判断是不是调用的main函数。这样以后调用的时候就可以防止不会多次调用 或者函数调用错误
if __name__=='__main__':
    main()
