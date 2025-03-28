import random

number_list = []

for i in range(9):
    row=[]
    for j in range(9):
        row.append(random.randint(0,99))
    number_list.append(row)
    
# max_num = max(number_list)
# print(number_list.index(max_num))
# max 함수는 1차원 리스트에서 작동함!

max_num=0
max_pose=(0,0)

for i in range(9):
    for j in range(9):
        if number_list[i][j] > max_num:
            max_num = number_list[i][j]
            max_pose = (i+1, j+1)

print(max_num) 
print(*max_pose)