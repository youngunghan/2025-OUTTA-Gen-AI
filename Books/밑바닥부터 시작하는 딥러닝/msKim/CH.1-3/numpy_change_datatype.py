import numpy as np

x = np.array([-1.0, 1.0, 2.0])
print(x)

y = x > 0
print(y)
y = y.astype(int)  # np.int 대신 int 사용
print(y)  # 출력: [0 1 1]
