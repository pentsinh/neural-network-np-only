import numpy as np
import matplotlib.pyplot as plt
from utils import load_and_preprocess_data, NeuralNetwork, DenseLayer


def main():
    """
    训练神经网络并保存权重
    """
    print("loading MNIST dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # 创建神经网络
    print("\nCreating neural network...")
    model = NeuralNetwork()
    
    # 添加全连接层
    # 输入层: 64个特征 (8x8像素图像展平)
    # 隐藏层1: 128个神经元
    # 隐藏层2: 64个神经元
    # 输出层: 10个神经元 (对应10个数字类别)
    model.add_layer(DenseLayer(input_size=64, output_size=128, activation='relu'))
    model.add_layer(DenseLayer(input_size=128, output_size=64, activation='relu'))
    # model.add_layer(DenseLayer(input_size=64, output_size=64, activation='sigmoid'))
    model.add_layer(DenseLayer(input_size=64, output_size=10, activation='softmax'))
    
    print("Starting training...")
    train_losses, val_accuracies = model.fit(
        X_train, y_train, X_val, y_val,
        epochs=100, learning_rate=0.01, batch_size=32
    )
    
    # 在测试集上评估
    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)
    print(f"\nTest set accuracy: {test_accuracy:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('../docs/training_curves.png')
    print("Training curves saved as training_curves.png")
    
    # 显示训练曲线
    plt.show()
    
    # 保存模型权重
    model.save_weights('../weights/model_weights.npz')
    
    # 显示预测结果
    print("\nPredictions for the first 10 test samples:")
    for i in range(10):
        print(f"True label: {y_test[i]}, Predicted label: {test_predictions[i]}")


if __name__ == "__main__":
    main()