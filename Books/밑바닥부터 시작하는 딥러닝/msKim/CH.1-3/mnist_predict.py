import numpy as np
from mnist import load_mnist
import pickle
from class_Function import *
import matplotlib.pyplot as plt

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y
    
x,t = get_data()
network = init_network()

accuracy_cnt = 0 # 정확하게 맞춘 개수를 저장할 변수
accuracies = [] # 누적 정확도 저장

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻음
    if p == t[i]: # 모델이 예측한 클래스(p)와 실제 정답이 같다면(t[i])
        accuracy_cnt += 1
    accuracies.append(accuracy_cnt / (i + 1))  # 현재까지의 정확도 기록

        
print("Accuracy:" + str((float(accuracy_cnt)) / len(x))) 
# 문자열과 숫자는 직접 연결할 수 없음! typeerror 방지 위해 str()로 변환함함

plt.figure(figsize=(8,5))
plt.plot(range(len(x)), accuracies, label = "Accuracy", color = "blue")
plt.xlabel("Numer of Samples")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Predictions")
plt.legend()
plt.grid()
plt.show()