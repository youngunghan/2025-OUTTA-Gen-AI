import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1) # 0에서 6까지 0.1 간격으로 생성함
y = np.sin(x)

plt.plot(x,y)
plt.show()