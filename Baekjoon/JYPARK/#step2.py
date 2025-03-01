#step2
#1 두수 비교
A, B = map(int, input().split())

if A > B:
    print('>')
elif A < B:
    print('<')
else:
    print('==')

#2 시험 성적
grade = int(input())

if grade < 60:
    print('F')
elif grade <70:
    print('D')
elif grade < 80:
    print('C')
elif grade < 90:
    print('B')
else:
    print('A')

#3 윤년
year = int(input())

if (year%4 ==0) and (year%100 !=0) :
    print(1)
elif year%400 == 0:
    print(1)
else:
    print(0)

#4 사분면 고르기
x = int(input())
y = int(input())

if x > 0:
    if y > 0:
        print(1)
    else:
        print(4)
else:
    if y > 0:
        print(2)
    else:
        print(3)

#5 알람시계
H, M = map(int, input().split())
total = H*60 + M 

if total >= 45:
    total -= 45
    print(total//60, total%60)
else:
    print(23, 60 +(M-45))

#6 오븐시계
A, B = map(int, input().split())
C = int(input())
total = A*60 + B + C

if total < 24*60 :
    print(total//60, total%60)
else:
    total -= 24*60
    print(total//60, total%60)

#7 주사위 세개
a, b, c = map(int, input().split())

if a == b == c:
    print(10000 + a*1000)
elif a == b or a == c:
    print(1000 + a*100)
elif b == c:
    print(1000 + b*100)
else:
    print(max(a,b,c)*100)