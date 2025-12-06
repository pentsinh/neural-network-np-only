import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DenseLayer:
    """
    全连接层实现
    """
    def __init__(self, input_size, output_size, activation='relu'):
        # 初始化权重和偏置
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        
        # 存储前向传播的值，用于反向传播
        self.input = None
        self.z = None  # 线性变换后的值
        self.output = None  # 激活函数后的值
        
    def forward(self, input_data):
        """
        前向传播
        """
        self.input = input_data
        # 线性变换: z = xW + b
        self.z = np.dot(input_data, self.weights) + self.biases
        
        # 激活函数
        if self.activation == 'relu':
            self.output = np.maximum(0, self.z)
        elif self.activation == 'softmax':
            # 数值稳定的softmax
            exp_z = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
            self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-np.clip(self.z, -250, 250)))  # 防止溢出
        elif self.activation == 'linear':  # 用于输出层的线性激活
            self.output = self.z
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
            
        return self.output
    
    def backward(self, grad_output, learning_rate):
        """
        反向传播
        """
        # 计算激活函数的导数
        if self.activation == 'relu':
            activation_derivative = (self.z > 0).astype(float)
            grad_z = grad_output * activation_derivative
        elif self.activation == 'softmax':
            # softmax的导数（在交叉熵损失下，简化处理）
            grad_z = grad_output
        elif self.activation == 'sigmoid':
            s = self.output
            activation_derivative = s * (1 - s)
            grad_z = grad_output * activation_derivative
        elif self.activation == 'linear':
            grad_z = grad_output
        else:
            grad_z = grad_output
        
        # 计算对权重和偏置的梯度
        grad_weights = np.dot(self.input.T, grad_z)
        grad_biases = np.sum(grad_z, axis=0, keepdims=True)
        
        # 计算对输入的梯度（传递给前一层）
        grad_input = np.dot(grad_z, self.weights.T)
        
        # 更新权重和偏置
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        
        return grad_input


class NeuralNetwork:
    """
    神经网络模型
    """
    def __init__(self):
        self.layers = []
        
    def add_layer(self, layer):
        """
        添加层到网络
        """
        self.layers.append(layer)
        
    def forward(self, x):
        """
        前向传播
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output, learning_rate):
        """
        反向传播
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        return grad
    
    def predict(self, x):
        """
        预测
        """
        output = self.forward(x)
        return np.argmax(output, axis=1)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size=32):
        """
        训练模型
        """
        n_samples = X_train.shape[0]
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # 随机打乱训练数据
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            # 分批处理训练数据
            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # 前向传播
                output = self.forward(X_batch)
                
                # 计算损失（交叉熵）
                loss = self.compute_loss(output, y_batch)
                epoch_loss += loss
                num_batches += 1
                
                # 计算输出层梯度（对于分类任务，使用交叉熵+softmax的梯度）
                grad_output = output.copy()
                grad_output[np.arange(len(y_batch)), y_batch] -= 1
                grad_output /= len(y_batch)  # 平均梯度
                
                # 反向传播
                self.backward(grad_output, learning_rate)
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            # 验证准确率
            val_predictions = self.predict(X_val)
            val_accuracy = np.mean(val_predictions == y_val)
            val_accuracies.append(val_accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return train_losses, val_accuracies
    
    def compute_loss(self, predictions, targets):
        """
        计算交叉熵损失
        """
        # 防止log(0)
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        # 交叉熵损失
        loss = -np.mean(np.log(predictions[np.arange(len(targets)), targets]))
        return loss

    def save_weights(self, filepath):
        """
        保存模型权重
        """
        weights_dict = {}
        for i, layer in enumerate(self.layers):
            weights_dict[f'layer_{i}_weights'] = layer.weights
            weights_dict[f'layer_{i}_biases'] = layer.biases
        np.savez(filepath, **weights_dict)
        print(f"Model weights saved to {filepath}")

    def load_weights(self, filepath):
        """
        加载模型权重
        """
        data = np.load(filepath)
        for i, layer in enumerate(self.layers):
            layer.weights = data[f'layer_{i}_weights']
            layer.biases = data[f'layer_{i}_biases']
        print(f"Model weights loaded from {filepath}")


def load_and_preprocess_data():
    """
    加载和预处理MNIST手写数字数据集
    """
    # 加载数据
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 进一步划分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test