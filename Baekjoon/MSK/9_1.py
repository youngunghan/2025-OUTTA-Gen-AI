# T = int(input())

# for i in range(T):
#     two_numbers = list(map(int, input().split()))
    
#     if two_numbers[1] % two_numbers[0] == 0:
#         print("factor")
#     elif two_numbers[0] % two_numbers[1] == 0:
#         print("multiple")
#     else : print("neither")

while True:
    two_numbers = list(map(int, input().split()))
    if two_numbers == [0,0]:
        break
    
    a,b = two_numbers[0], two_numbers[1]
    
    if a%b == 0:
        print("multiple")
    elif b%a == 0:
        print("factor")
    else:
        print("neither")