import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# XOR 데이터셋
data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
labels = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 다층 퍼셉트론 모델 정의
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(2, 2)  # 은닉층
        self.output = nn.Linear(2, 1)  # 출력층
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        hidden_output = self.activation(self.hidden(x))
        final_output = self.activation(self.output(hidden_output))
        return hidden_output, final_output

# 모델 생성
model = MLP()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 학습 과정
for epoch in range(30000):
    optimizer.zero_grad()
    hidden_output, output = model(data)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 5000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 학습 후 중간과정 시각화 (꺾은선 그래프)
print("\nFinal Predictions with Line Graph Visualization:")
with torch.no_grad():
    hidden_outputs, final_outputs = model(data)
    
    hidden_outputs = hidden_outputs.numpy()
    final_outputs = final_outputs.numpy()
    
    input_labels = ['[0,0]', '[0,1]', '[1,0]', '[1,1]']
    
    # 꺾은선 그래프 그리기
    plt.figure(figsize=(10, 6))
    
    for i in range(4):
        y_values = [hidden_outputs[i][0], hidden_outputs[i][1], final_outputs[i][0]]
        plt.plot(['Hidden1', 'Hidden2', 'Output'], y_values, marker='o', label=f'Input {input_labels[i]}')
    
    plt.ylim(0, 1)
    plt.title('Activation Values per Input')
    plt.ylabel('Activation Value')
    plt.xlabel('Layer Neuron')
    plt.legend()
    plt.grid(True)
    plt.show()
