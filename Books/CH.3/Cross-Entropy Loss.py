import numpy as np
import matplotlib.pyplot as plt

# 실제 라벨
true_label = [1, 0, 0]  # 예제에서는 실제 라벨이 [1, 0, 0]로 설정됨 (1번째 클래스가 정답)

# 모델의 예측 확률 (점진적으로 개선)
predictions = np.linspace(0.1, 0.9, 100)  # 0.1부터 0.9까지 100개의 예측 확률 생성
losses = -np.log(predictions)  # 교차 엔트로피 손실 계산: -log(예측 확률)

plt.figure(figsize=(6, 4))  # 그래프 크기를 가로 6, 세로 4로 설정
plt.plot(predictions, losses, label="Cross-Entropy Loss")  # 예측 확률에 따른 손실 값 그래프
plt.title("Cross-Entropy Loss")  # 그래프 제목 설정
plt.xlabel("Predicted Probability for True Label")  # X축 라벨: 올바른 클래스에 대한 예측 확률
plt.ylabel("Loss")  # Y축 라벨: 손실 값

# 예측 확률이 0.7인 지점을 강조
plt.axvline(x=0.7, color='r', linestyle='--', label="Good Prediction (0.7)")  
# 예측 확률이 0.7일 때를 수직선으로 표시 (좋은 예측으로 간주)

plt.legend()  # 범례 추가
plt.show()  # 그래프 표시
