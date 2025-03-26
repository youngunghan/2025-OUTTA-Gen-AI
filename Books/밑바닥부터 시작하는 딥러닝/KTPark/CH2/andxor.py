import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.where(x > 0, 1, 0)

def perceptron(x, w, b):
    return step_function(np.dot(x, w.T) + b)

# NAND 게이트
w_nand = np.array([[-1, -1]])
b_nand = np.array([1.5])

# OR 게이트
w_or = np.array([[1, 1]])
b_or = np.array([-0.5])

# AND 게이트
w_and = np.array([[1, 1]])
b_and = np.array([-1.5])

# XOR 게이트 (중간 출력 반환)
def xor_gate_trace(x):
    nand_out = perceptron(x, w_nand, b_nand)
    or_out = perceptron(x, w_or, b_or)
    combined = np.hstack((nand_out, or_out))
    final_out = perceptron(combined, w_and, b_and)
    return nand_out[0], or_out[0], final_out[0]

# 입력 데이터
x_data = np.array([[0,0], [0,1], [1,0], [1,1]])
input_labels = ['[0,0]', '[0,1]', '[1,0]', '[1,1]']

# 결과 저장용 리스트
results = []

# 각 입력에 대한 중간 및 최종 출력 수집
for x in x_data:
    nand_val, or_val, final_val = xor_gate_trace(x)
    results.append([nand_val, or_val, final_val])
    print(f'입력: {x}, NAND: {nand_val}, OR: {or_val}, 출력: {final_val}')

# 그래프 시각화
results = np.array(results)

plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(['NAND', 'OR', 'AND_Output'], results[i], marker='o', label=f'Input {input_labels[i]}')

plt.ylim(-0.1, 1.1)
plt.title('XOR Gate Intermediate Outputs (Perceptron Logic)')
plt.ylabel('Output Value (0 or 1)')
plt.xlabel('Gate')
plt.grid(True)
plt.legend()
plt.show()
