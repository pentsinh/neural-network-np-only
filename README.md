# MNIST手写数字识别神经网络 - 模块化版本

## 项目概述
这是一个模块化的MNIST手写数字识别神经网络实现，将原始代码拆分为独立的模块，便于维护和重用。该实现包含训练、测试和工具模块，并支持权重的保存和加载。

## 文件结构
- `train.py`: 模型训练脚本，包含完整的训练流程和权重保存功能
- `test.py`: 模型测试脚本，用于加载已训练权重并进行预测
- `utils.py`: 工具模块，包含神经网络核心类、层定义和数据预处理函数
- `model_weights.npz`: 训练后保存的模型权重文件（运行train.py后生成）

## 模块说明

### 1. utils.py - 工具模块
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

### 2. train.py - 训练模块
- 创建神经网络模型
- 执行训练流程
- 保存训练后的权重到`model_weights.npz`
- 生成训练曲线图`training_curves.png`
- 训练结束后显示loss曲线

### 3. test.py - 测试模块
- 加载已训练的权重
- 在测试集上自动评估模型性能
- 显示测试集准确率和前10个样本的预测结果
- 显示详细分类报告和混淆矩阵
- 完全自动化的测试过程，无需用户交互

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

## 技术特点
- 模块化设计，代码结构清晰
- 支持权重保存和加载
- 数值稳定的Softmax实现
- 防止梯度爆炸的梯度裁剪
- 批量训练支持
- 交叉熵损失函数

## 使用场景
- 模型训练和评估
- 模型权重重用
- 快速预测验证
- 作为更复杂项目的参考实现