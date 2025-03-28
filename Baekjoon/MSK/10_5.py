N = int(input())
x_dot = []
y_dot = []

for i in range(N):
    x, y = map(int, input().split())
    x_dot.append(x)
    y_dot.append(y)

w = max(x_dot) - min(x_dot)
h = max(y_dot) - min(y_dot) 
area = w * h
print(area)