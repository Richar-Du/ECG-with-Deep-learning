# ECG-with-Deep-learning

本仓库分为如下两部分：
- 文档
  - 深度学习模型方面：记录了我们小组从零开始，一步步摸索搭建深度学习模型的过程，解释了我们每一阶段选择某种特定模型的理由，也对我们的改进思路进行了阐述；
  - 数据融合方面：从众多数据库中挑选出4个较为合适的进行了详细介绍，包括融合的详细过程；
  - 硬件部署：使用raspberry zero w,AD8232心电模块,PCF8591模数转换模块，采集人体心电数据，在树莓派上进行实时分类，实现边缘计算。
- 代码
  - train.py
  - model:
    - CNN.py
    - CNNLSTM.py
    - SENetLSTM.py
  - save
    - CNN
    - CNNLSTM
    -SENetLSTM
  - tensorboard
    - CNN20201201-103452
    - CNNLSTM20201224-153049
    - SENetLSTM20210108-154007
  - tflite
    - model.tflite
  - checkpoint
