import numpy as np
from utils import load_and_preprocess_data, NeuralNetwork, DenseLayer
from sklearn.metrics import classification_report, confusion_matrix


def main():
    """
    加载已训练的模型权重并进行测试，输出准确率
    """
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data()
    
    # 创建神经网络结构（与训练时相同）
    model = NeuralNetwork()
    
    # 添加全连接层（必须与训练时完全一致）
    model.add_layer(DenseLayer(input_size=64, output_size=128, activation='relu'))
    model.add_layer(DenseLayer(input_size=128, output_size=64, activation='relu'))
    model.add_layer(DenseLayer(input_size=64, output_size=10, activation='softmax'))
    
    # 加载已训练的权重
    try:
        model.load_weights('./model_weights.npz')
    except FileNotFoundError:
        print("Can't find the model weights file.Check if the file exists.")
        return
    
    # 在测试集上评估
    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)
    
    # 输出准确率
    print(f"{test_accuracy:.4f}")


if __name__ == "__main__":
    main()