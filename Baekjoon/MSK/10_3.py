x_dot = []
y_dot = []

for _ in range(3):
    x, y = map(int, input().split())
    x_dot.append(x)
    y_dot.append(y)

# x좌표 중 하나만 나온 것 찾기
for i in x_dot:
    if x_dot.count(i) == 1:
        x_result = i

# y좌표 중 하나만 나온 것 찾기
for j in y_dot:
    if y_dot.count(j) == 1:
        y_result = j

print(x_result, y_result)

