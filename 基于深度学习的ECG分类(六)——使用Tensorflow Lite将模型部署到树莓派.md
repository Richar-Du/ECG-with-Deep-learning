## 基于深度学习的ECG分类(六)——使用Tensorflow Lite将模型部署到树莓派

部署之前我们首先要明确，Tensorflow模型实质上就是一种数据结构，其内部有自身的逻辑运算顺序，把一个符合输入格式的数据送到模型当中去，模型就会按照它自身的数据结构和逻辑运算计算出一个结果，反馈给我们。因此我们只需要想办法把这个数据结构存储到边缘设备当中，就能实现所谓的“边缘计算”。而Tensorflow Lite就是在边缘设备中执行Tensorflow模型的工具。

> 注意Tensorflow Lite只是一个模型的执行工具，而不是一个模型训练工具。

下面按照一整套流程介绍如何将Tensorflow模型部署到树莓派。

- 转化模型：

  Tensorflow Lite之所以能够在资源较小的嵌入式设备中高效运算模型，很大程度上是使用了一种特殊的数据结构去存储模型。因此在使用Tensorflow Lite之前，必须先将模型转化成Tensorflow Lite的格式。方式一共有两种：使用python API和命令行，此处我们用第一种，也是Tensorflow官方推荐的。

  在模型转化之前有必要看一下Tensorflow Lite支持的Tensorflow的运算(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/guide/ops_compatibility.md)。如果要转换的模型包含不受支持的操作，则可以使用TensorFlow Select包含来自TensorFlow的操作。但是这将产生更大的二进制文件。

```python
import tensorflow as tf
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```
- 采集ECG数据：
  
  我们利用AD8232模块以360HZ的采样频率采集10s ECG数据，然后再通过PCF85591模块将ECG模拟信号转换为ECG数字信号
  
  在树莓派上，我们通过smbus包来控制I2C总线，获取ECG数据并进行识别，主要步骤如下：
  
  - 打开树莓派的I2C总线
  - 向PCF8591发送命令，选择输入口
    - 因为PCF8591有四个输入口，所以我们要告诉PCF8591监听哪一个输入口
  - 读取PCF8591传来的数据
  
```python
# read the ECG data
# The total amount of data is freq * duration
def get_ecg(freq,duration):
	ecgs = []
	address = 0x48
	A0 = 0x40
	A1 = 0x41
	A2 = 0x42
	A3 = 0x43
	bus = smbus.SMBus(1)
	bus.write_byte(address,A2)
	for i in range(int(duration*freq)):
		value = bus.read_byte(address)
		ecgs.append(value)
		time.sleep(1.0/freq)
	return ecgs
  
ecg_signals = get_ecg(360,10)
```

- 使用模型进行推断

  所谓“推断”就是通过模型运行数据以获得预测的过程。它需要一个模型，一个解释器和输入数据。具体流程有如下几步：

  - 加载模型：把.tflite文件加载到内存中，其内部有计算图；
  - 转化数据：把数据格式改成适用于模型的形式；
  - 运行推断：使用Tensorflow Lite的API执行模型，包括：构建解释器、分配张量等；
  - 解释输出：例如在心电数据分类中，模型的输出是几个概率，需要我们映射到一个类别。

  基于模型的推断可以在各种平台上运行，因为我们这里用的树莓派Linux系统，因此最好的方式就是用python来运行。

  > 解释器：TensorFlow Lite解释器是一个库，该库获取模型文件，对输入的数据执行模型中定义好的计算，并输出数据。

  如果只是想用python运行Tensorflow Lite格式的模型，最快的方式就是安装Tensorflow Lite的解释器，而非整个Tensorflow包，这可以节省很大的空间。

  创建虚拟环境（没有创建软链接时需要写完整的virtualenv路径）

  ```shell
  /usr/bin/virtualenv/virtualenv -p /usr/bin/python3.5 mytf2
  ```

  下载tflite或者tensorflow（如果不在乎空间大小可以将tensorflow完全下载下来）

  在树莓派上部署只需要做模型的推断，即前向传播即可：

  ```python
  import numpy as np
  import tensorflow as tf

  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Test the model on random input data.
  input_shape = input_details[0]['shape']
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  
  # Tead the ECG data
  interpreter.set_tensor(input_details[0]['index'],ecg_signals)
  
  # Predict
  interpreter.invoke()
  output_data = interpreter.get_tensor(output_details[0]['index'])
  print(output_data)
  ```


