# MNIST手写数字识别神经网络

## 项目概述
这是一个模块化的MNIST手写数字识别神经网络实现,包含训练、测试和工具模块，并支持权重的保存和加载。

## 课程报告
[基于numpy实现的全连接层神经网络的MNIST分类任务](./docs/基于numpy实现的全连接层神经网络的MNIST分类任务.pdf)

## 快速开始
```bash
cd ./scripts
python ./train.py #训练
python ./test.py #测试
```

## 模块说明

###  [utils.py](./scripts/utils.py) - 工具模块
- **DenseLayer类**: 全连接层的实现，包含前向传播和反向传播
  - `forward(input_data)`: 前向传播计算
  - `backward(grad_output, learning_rate)`: 反向传播更新权重
- **NeuralNetwork类**: 神经网络模型管理类
  - `add_layer(layer)`: 添加层到网络
  - `forward(x)`: 前向传播
  - `backward(grad_output, learning_rate)`: 反向传播
  - `predict(x)`: 预测函数
  - `fit(X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size)`: 训练模型
  - `save_weights(filepath)`: 保存模型权重
  - `load_weights(filepath)`: 加载模型权重
- **load_and_preprocess_data()**: 加载并预处理MNIST数据集

### [train.py](./scripts/train.py) - 训练模块
- 创建神经网络模型
- 执行训练流程
- 保存训练后的权重到`model_weights.npz`
- 生成训练曲线图`training_curves.png`
- 训练结束后显示loss曲线

### [test.py](./scripts/test.py) - 测试模块
- 加载已训练的权重
- 在测试集上自动评估模型性能
- 显示测试集准确率和前10个样本的预测结果


## 网络结构
- **输入层**: 64个神经元 (对应8x8像素图像展平)
- **隐藏层1**: 128个神经元，ReLU激活函数
- **隐藏层2**: 64个神经元，ReLU激活函数
- **输出层**: 10个神经元 (对应10个数字类别)，Softmax激活函数

## 训练参数
- **训练轮数**: 100 epochs
- **学习率**: 0.01
- **批量大小**: 32
- **验证集比例**: 20% (从训练集中划分)
- **测试集比例**: 20%

## 使用方法

### 1. 训练模型
```bash
python train.py
```
- 该命令将训练模型并保存权重到`model_weights.npz`
- 同时生成训练曲线图`training_curves.png`

### 2. 测试模型
```bash
python test.py
```
- 该命令将加载已保存的权重并评估模型性能
- 显示测试集准确率和前10个样本的预测结果

