1. # GRU

   GRU(Gated Recurrent Neural Network，门控循环单元)，与LSTM一样，也是为了解决普通RNN无法进行长程记忆的问题。可以说，GRU是LSTM的一种变体。那么，GRU的结构是什么样的呢？GRU和LSTM相比有哪些不同又有那些共同点呢？下面我就来为大家一一介绍。

   ## 1 LSTM简介

   我们要研究鸡蛋，得先去了解母鸡。LSTM就是GRU的母鸡，所以我们先来简单看一下LSTM是个什么东西。

   ### 1.1 RNN与LSTM

   RNN（Recurrent Neural Network）是一种具有大量环路的神经网络，环路的存在使得RNN比普通的前馈神经网络具有更好的记忆功能。但是因为人工神经元在处理信息时存在信号的衰减，RNN的记忆能力仍然不足以应对真实序列中的长程依赖特性。

   下图是一个简单的RNN网络结构，红色的部分是同层间的连接。除了同层间的连接以外，RNN和普通的前馈神经网络没有什么不同。

   <img src="figure\image-20201130204749687.png" alt="image-20201130204749687" style="zoom:40%;" />

   为了解决RNN无法进行长程记忆的问题，尤根·斯提姆哈勃提出了LSTM(Long Short Term Memory，长短时记忆)。

   ### 1.2 LSTM的结构

   这是一个LSTM结构的简单示意图

   <img src="figure\image-20201202211432502.png" alt="image-20201202211432502" style="zoom:33%;" />

   如上图，我们可以把LSTM的结构简要概括为**同外界交流的三要素**以及**对三要素进行控制的三个门**

   - ##### 同外界交流的三要素

     - 来自同一时间步上一层神经元的数据**输入**$x(t)$
     - 来自同一层神经元上一时间步的数据**输入**$h(t-1)$
     - LSTM单元对外界的数据**输出**$h(t)$

   - ##### 控制三要素的三个门

     - 控制输入$x(t)$的**输入门**$i(t)$
     - 控制输入$h(t-1)$**遗忘门**$f(t)$
     - 控制输出$h(t)$的**输出门**$o(t)$

   ##### 进一步，一个LSTM单元的运作可以写成如下动力学方程

   ![image-20201130214858720](figure\image-20201130214858720.png)

   ##### 相应的运作过程如下图所示：

   <img src="figure\image-20201202210935078.png" alt="image-20201202210935078" style="zoom:40%;" />

   **1** 来自同层上一个时间步的数据$h(t-1)$和来自上一层同一时间步的数据$x(t)$经过一个tanh激活函数得到$g(t)$

   **2** $g(t)$经过输入门$i(t)$得到$i(t)*g(t)$

     本单元历史隐藏态$c(t-1)$经过遗忘门$f(t)$得到$f(t)*c(t-1)$

   **3** $i(t)*g(t)$和$f(t)*c(t-1)$相加得到$c(t)$

   **4** $c(t)$保存为本单元隐藏状态

     $c(t)$经过一个$tanh$激活函数然后再经过输出门$o(t)$得到本单元的输出$h(t)$

   ## 2 GRU的结构

   这是一个GRU结构的简单示意图：

   <img src="figure\image-20201202214209095.png" alt="image-20201202214209095" style="zoom:40%;" />

   - GRU**同外界交流的三个元素**与LSTM相同
     - 来自同一时间步上一层神经元的数据**输入**$x(t)$
     - 来自同一层神经元上一时间步的数据**输入**$h(t-1)$
     - GRU单元对外界的数据**输出**$h(t)$

   相比于LSTM的三个门：输入门$i(t)$、遗忘门$f(t)$、输出门$o(t)$

   - GRU仅有**两个门**

     - 对来自上一时间步的$h(t-1)$进行选择性遗忘的**重置门**$r(t)$
       - $\tilde h(t) = tanh(W·[r(t)*h(t-1),x(t)])$
     - 权衡旧数据$h(t-1)$和新数据$\tilde h(t)$的**更新门**$z(t)$
       - $h(t) = (1-z(t))*h(t-1)+z(t)*\tilde h(t)$

   - ##### 通俗来讲

     - **重置门** $r(t)$ 的运作：GRU综合**过去的"认识"**$h(t-1)$和**来自上一层神经元的输入**$x(t)$得到了**现在的新“认识”**$\tilde h(t)$
     - **更新门** $z(t)$ 的运作：GRU对**过去的"认识"**$h(t-1)$和**现在的新“认识”**$\tilde h(t)$进行加权，作为输出

   ##### 一个GRU单元的运作可以写成如下动力学方程

   <img src="figure\image-20201202215111415.png" alt="image-20201202215111415" style="zoom: 80%;" />

   ##### 相应的运作过程如下

   <img src="figure\image-20201202215322574.png" alt="image-20201202215322574" style="zoom:60%;" />

   ​

   ## 3 GRU vs LSTM

   - ##### 方程

     - ![image-20201202220249427](figure\image-20201202220249427.png)

   - ##### 运作

     - ![image-20201202220350741](figure\image-20201202220350741.png)

   ##### 总的来说

   - GRU更简单，计算量更小，速度更快，更容易去创建一个大的网络
   - LSTM更强大，更灵活

   通俗来讲，考虑计算量和表现，GRU与LSTM相比，就好像是两个手的孙悟空(GRU)和六个手的哪吒(LSTM)。

   （我们假设哪吒三个头的算力和孙悟空一个头一样，这里的头就好比是我们的电脑或服务器，手就好比是我们的神经网络）

   <img src="figure\image-20201202220843646.png" alt="image-20201202220843646" style="zoom:67%;" />

   计算量上，哪吒有六个手，比孙悟空多，计算量自然更大。

   表现上，则是在某些任务具备优越性，而在另外一些任务表现不出优越性。比如哪吒有了六个手，在搬砖上自然就可以一次搬六个，这里有优越性；但是即便有六个手，只要哪吒的算力和孙悟空是一样的，那么他在解数学题上相对孙悟空便没有多大优越性。

   ## 4 GRU的实现

   - ##### return_sequences

     - Boolean. Whether to return the last output in the output sequence, or the full sequence. Default: `False`.

   - ##### return_state

     - Boolean. Whether to return the last state in addition to the output. Default: `False`.

   ```python
   # 仅返回最后的隐状态，即（batch_size,units)
   inputs = tf.random.normal([32, 10, 8])
   gru = tf.keras.layers.GRU(4)
   output = gru(inputs)
   print(output.shape)
   (32, 4)

   # 返回所有时刻的隐状态以及最后的隐状态，即（batch_size,time_steps,units）,（batch_size,units)
   gru = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)
   whole_sequence_output, final_state = gru(inputs)
   print(whole_sequence_output.shape)
   (32, 10, 4)
   print(final_state.shape)
   (32, 4)
   ```

   ## 5 参考

   1. 《深度学习原理与Pytorch实战》集智俱乐部
   2. https://keras.io/api/layers/recurrent_layers/gru/
   3. https://blog.csdn.net/qq_29831163/article/details/89929573
   4. https://keras.io/api/layers/recurrent_layers/gru/