---
title: 使用Docker部署模型
author: 都一凡
date: 2020-11-18 16:12:00 +0800
categories: [blog]
tags: [Deep learning]
image: D:\Myblog\Richar-Du.github.io\assets\img\sample\avatar.jpg
pin: true
---

模型训练完毕后，下一步就是部署。一个常用的情景就是在服务器上提供一个 API，用户如果有某个需求，就向服务器对应的 API 发送特定格式的请求，服务器收到请求数据后通过模型进行计算，并返回结果。这属于在线部署模型，虽然我们最终的目标是边缘计算，但是在线部署是“端-管-云”架构的重要体现，之前的项目中只用过边缘计算的我也想尝试一下这种方式。

在工业界常用的部署方式就是借助Docker，服务器上已经安装了Docker，通过运行以下命令来获取最新的TensorFlow Serving Docker映像，这里面有tensorflow自带的一个小模型。注意这个拉取下来后不是直接放在当前目录的，而是docker默认存储的路径：

```shell
docker pull tensorflow/serving
```

下面是tensorflow官网自带的一个例子：

```shell
docker run -p 8501:8501 \
	--mount type=bind,\
source=/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu,\
	target=/models/half_plus_two \
 	-e MODEL_NAME=half_plus_two -t tensorflow/serving &
```

其中参数含义如下：

```shell
--mount：   表示要进行挂载
source：    指定要运行部署的模型地址， 也就是挂载的源，这个是在宿主机上的模型目录
target:     这个是要挂载的目标位置，也就是挂载到docker容器中的哪个位置，这是docker容器中的目录
-t:         指定的是挂载到哪个镜像
-p:         指定主机到docker容器的端口映射
docker run: 启动这个容器并启动模型服务
&:			后台运行，也可以用-d

综合解释：
将source目录中的例子模型，挂载到-t指定的docker容器中的target目录，并启动
```

跑一遍tensorflow官方github的例子，它的模型是0.5*x+2，这里注意一下模型的路径结构，因为后面要替换成我们自己的模型时，保存的模型路径也是这样子的。

下面是把数据POST到服务器8501端口上：

```shell
curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://localhost:8501/v1/models/half_plus_two:predict
```

最终结果如下：

```shell
{ "predictions": [2.5, 3.0, 4.5] }
```

官网的例子已经跑通了，下面换成我们自己的模型，我们的模型保存在了`/home/group5/ECG/save/CNN`：

```shell
docker run -p 8501:8501 \
	--mount type=bind,\
	source=/home/group5/ECG/save/CNN,\
	target=/models/ECG2 \
	-e MODEL_NAME=ECG2 -t tensorflow/serving -- &
```

> 一定要注意source中的文件结构要和官网给的模型的文件结构一致，target是docker的文件目录而不是系统的文件目录，MODEL_NAME要和target中模型的名字保持一致，否则会找不到文件

向服务器发送如下心跳样本：

```shell
curl -d '{"instances": [[[-0.145],[-0.145],[-0.145],...]]}' -X POST http://localhost:8501/v1/models/half_plus_two:predict
```

最后返回的结果如下：

```shell
{
    "predictions": [[1.3092547e-24, 1.61444859e-13, 7.92131863e-14, 1.0, 6.66244249e-19, 9.92791136e-13, 2.16822859e-15]
    ]
}
```

可以看出，属于第四类的概率是1，因此该样本应该被分为第四类。

整个过程由于对docker不是很熟悉，因此摸索了很多，下面是一些常用命令：

```shell
docker images		# 查看镜像
docker container ls		# 查看正在运行的容器
docker container stop [container-ID]		# 停止容器
docker run -d -p 81:80	ngix	# 有了ngix镜像之后，运行一个docker容器，-d表示后台运行，-p表示端口映射，此处是把容器的80映射到主机的81
docker exec -it [container-ID] bash			# 进入容器，可以对容器内部的文件进行修改
docker rm -f [container-ID]			# 删除一个docker
docker commit [container-ID] [NewImage]   # 把containe-ID复制成一个新的容器，镜像叫NewImage
docker run -d -p 90:80 [NewImage] --name [ContainerName]	# 为这个新的容器指定端口映射，指定新容器名字为ContainerName
```

`docker container run`命令是新建容器，每运行一次，就会新建一个容器。同样的命令运行两次，就会生成两个一模一样的容器文件。如果希望重复使用容器，就要使用`docker container start`命令，它用来启动已经生成、已经停止运行的容器文件。

```shell
$ docker container start [containerID]
```
> -d参数很重要，它是让docker在后台运行，不阻滞命令行，很多不知所措的情况就是没有让docker在后台运行。

关于网络显示拒绝连接的一些故障排除：

```shell
env|grep -I proxy
```

没有显示使用代理。

```shell
lsof -i:8051
```

显示端口没有被占用

```shell
firewall-cmd --state
```

显示防火墙没有开启
