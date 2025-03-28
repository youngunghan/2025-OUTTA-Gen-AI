# number : 변환하려는 숫자, base = 진법
number, base = map(int, input().split())
result = ''
arr = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

while number:
    result += str(arr[number%base])
    number //= base

print(result[::-1])