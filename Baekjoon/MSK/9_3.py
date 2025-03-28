import math

# 약수들의 집합을 구하는 함수 - 매커니즘 이해 못함
def find_divisors(n):
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)  # i를 약수 목록에 추가
            if i != 1 and i != n // i:  # 자기 자신은 제외하고, 중복된 약수도 제외
                divisors.append(n // i)  # n을 i로 나누어서 나온 값도 약수에 추가
    divisors.sort()  # 오름차순으로 정렬
    return divisors

while True:
    num = int(input())
    if num == -1:
        break
    
    divisors = find_divisors(num)
    divisor_sum = sum(divisors)  
    
    if divisor_sum == num:
        print(f"{num} = {' + '.join(map(str, divisors))}")
    else:
        print(f"{num} is Not perfect.")  
        
        
# f-string을 사용한 출력 : 중괄호 {} 안에 표현식을 넣으면 그 값이 문자열로 변환되어 출력됨
# join() 함수: 리스트의 모든 요소를 문자열로 결합하고, 지정된 구분자('+')로 연결해 하나의 문자열 만듦듦
# map(str, divisors) : divisors 리스트에 있는 모든 요소를 문자열로 변환함


    
    