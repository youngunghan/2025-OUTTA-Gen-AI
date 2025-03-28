import math

first_num = int(input())
second_num = int(input())

def is_prime(n):
    if n < 2:  # 1 이하의 숫자는 소수가 아님
        return False
    
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:  # 나누어떨어지는 수가 있으면 소수가 아님
            return False
    return True

total = []

for i in range(first_num, second_num+1):
    if is_prime(i):
        total.append(i)

if total==[]:
    print(-1)
else:
    print(sum(total))
    print(min(total))
    

        
