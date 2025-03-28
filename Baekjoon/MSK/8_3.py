''' 입력을 한 번에 받는 방식
T = int(input())
coins = [25, 10, 5, 1]
give_back = list(map(int, input().split()))

for i in give_back:
    result = []
    for coin in coins:
        result.append(i//coin)
        i %= coin
        
    print(*result) '''
    
# 입력을 각 줄마다 받기
T = int(input())
coins = [25, 10, 5, 1]
list_give_back = []

for i in range(T):
    give_back = int(input())
    list_give_back.append(give_back)
    
for i in list_give_back:
    result = []
    for coin in coins:
        result.append(i//coin)
        i %= coin
        
    print(*result)
    