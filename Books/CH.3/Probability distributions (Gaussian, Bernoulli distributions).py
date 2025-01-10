import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli

# 가우시안 분포
x = np.linspace(-4, 4, 1000)  # -4부터 4까지 1000개의 점 생성 (X 축의 범위 설정)
mean, std = 0, 1  # 평균(mean)과 표준편차(standard deviation)를 각각 0과 1로 설정
pdf = norm.pdf(x, mean, std)  # 지정된 평균과 표준편차를 사용해 가우시안 확률밀도함수(PDF) 계산

plt.figure(figsize=(10, 4))  # 전체 그래프 크기를 설정 (가로 10, 세로 4)

# 가우시안 분포 그래프
plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 서브플롯
plt.plot(x, pdf, label="Gaussian PDF")  # 가우시안 PDF를 그래프로 표시
plt.title("Gaussian Distribution")  # 그래프 제목 설정
plt.xlabel("x")  # X축 라벨 설정
plt.ylabel("Density")  # Y축 라벨 설정
plt.legend()  # 범례 추가

# 베르누이 분포
p = 0.7  # 성공 확률(success probability)을 0.7로 설정
x = [0, 1]  # 베르누이 분포에서 가능한 값 (0과 1)
pmf = bernoulli.pmf(x, p)  # 지정된 확률 p를 사용해 베르누이 확률질량함수(PMF) 계산

# 베르누이 분포 그래프
plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 서브플롯
plt.bar(x, pmf, color='orange', label="Bernoulli PMF")  # PMF 값을 막대그래프로 표시
plt.title("Bernoulli Distribution")  # 그래프 제목 설정
plt.xlabel("x")  # X축 라벨 설정
plt.ylabel("Probability")  # Y축 라벨 설정
plt.xticks([0, 1])  # X축 눈금을 0과 1로 설정
plt.legend()  # 범례 추가

plt.tight_layout()  # 서브플롯 간의 간격 조정
plt.show()  # 모든 그래프를 화면에 표시
