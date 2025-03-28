total = int(input())  # 과목 개수 입력
original_score = list(map(int, input().split() [:total]))  # 점수 리스트로 변환

max_score = max(original_score)  # 가장 높은 점수 찾기
new_numbers = []

for i in original_score:
    new_numbers.append(i/max_score*100)

# 평균 계산
average = sum(new_numbers) / total  

print(average)  # 변환된 점수들의 평균 출력


# for문 대신 점수를 변환하여 new_numbers 리스트에 저장 (리스트 컴프리헨션 활용)
# new_numbers = [(i / max_score * 100) for i in original_score]