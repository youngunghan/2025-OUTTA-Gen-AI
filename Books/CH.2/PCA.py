import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# 예시 데이터 생성
np.random.seed(0)  # 난수를 생성할 때 동일한 결과를 얻기 위해 시드 설정
data = np.random.randn(100, 3)  # 100개의 샘플을 가진 3차원 랜덤 데이터 생성

# PCA 적용하여 2차원으로 축소
pca = PCA(n_components=2)  # PCA 객체 생성, 축소할 차원을 2로 설정
principal_components = pca.fit_transform(data)  # PCA를 사용해 3차원 데이터를 2차원으로 변환

# 3차원 원본 데이터 시각화
fig = plt.figure(figsize=(12, 6))  # 그래프 크기 설정

# 원본 데이터 (3D)
ax1 = fig.add_subplot(121, projection='3d')  # 3D 그래프를 위한 서브플롯 추가
ax1.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', alpha=0.7)  # 3차원 데이터 산점도
ax1.set_title('Original 3D Data')  # 그래프 제목 설정
ax1.set_xlabel('X')  # X축 라벨
ax1.set_ylabel('Y')  # Y축 라벨
ax1.set_zlabel('Z')  # Z축 라벨

# PCA 결과 (2D)
ax2 = fig.add_subplot(122)  # 2D 그래프를 위한 서브플롯 추가
ax2.scatter(principal_components[:, 0], principal_components[:, 1], c='r', alpha=0.7)  # 2차원 데이터 산점도
ax2.set_title('PCA Result in 2D')  # 그래프 제목 설정
ax2.set_xlabel('Principal Component 1')  # 첫 번째 주성분 축 라벨
ax2.set_ylabel('Principal Component 2')  # 두 번째 주성분 축 라벨

plt.tight_layout()  # 그래프 간격 조정
plt.show()  # 그래프 출력
