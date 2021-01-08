记录了第一次跑深度学习时遇到的一些问题，并不是完整的流程，而是初学者容易犯的一些错误，比较简单基础，但是以后还会再用到。

- MIT-BIH心电数据库极度地不平衡，而不管是深度学习还是机器学习，他们能够较好地拟合数据的一个假设就是数据集是平衡分布的。因此为了更准确地评估深度学习模型的表现，采取分层抽样的方式，保证训练集和测试集的数据分布是一样的：

  ```python
  sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, train_size=0.9,random_state=0)   # n_split=1就只有一九分，如果需要交叉验证，把训练和测试的代码放到for循环里面就可以
  sss.get_n_splits(X, Y)
  for train_index, test_index in sss.split(X, Y):
      print("TRAIN:", train_index, "TEST:", test_index)
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = Y[train_index], Y[test_index]
  ```

- 由于心电数据的标签是字符串，为了使模型能正常运行，需要转成float：

  - tensorflow.keras.utils 中的 to_categorical函数只能转化文本型的数字，对于本问题不能使用

  - sklearn.preprocessing中的LabelEncoder：

    ```python
    from sklearn import preprocessing
    AAMI=['N','L','R','V','A','+','/']
    le=preprocessing.LabelEncoder()
    le=le.fit(AAMI)
    Y=le.transform(Y)
    ```

- tensorflow模型输入数据的维数必须是3维，类似于图像数据。但是在该项目的数据集中，训练集中的X是二维的矩阵，因此必须先扩充成三维。

  ```python
  X=np.expand_dims(X,axis=2)	# 一共有(0,1,2)三个通道，axis=2表示在2通道上扩充
  ```

- tensorflow模型初始化的时候不能直接送入数据，必须先初始化张量tensor，在模型调用fit方法的时候再输入数据。

  ```python
  input_ecg=tf.keras.layers.Input(shape=(3600,1))
  label_ecg=CNN1d(input_ecg)
  model=Model(input_ecg,label_ecg)
  ```

- 在 `tf.keras` 中，有两个交叉熵相关的损失函数 `tf.keras.losses.categorical_crossentropy` 和 `tf.keras.losses.sparse_categorical_crossentropy` 。其中 sparse 的含义是，真实的标签值 `y_true` 可以直接传入 int 类型的标签类别。具体而言：

  ```python
  loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
  ```

  与

  ```python
  loss = tf.keras.losses.categorical_crossentropy(
      y_true=tf.one_hot(y, depth=tf.shape(y_pred)[-1]),
      y_pred=y_pred
  )
  ```

  的结果相同。

  > 这两个函数用的时候一定要注意，不细心就会用混。

- tensorboard的使用