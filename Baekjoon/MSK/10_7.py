def division(x, y, z):
  
    if 2 * max(x,y,z) >= sum(x,y,z):
        return "Invalid"
    if x == y == z:
        return "Equilateral"
    elif x == y or y == z or x == z:
        return "Isosceles"
    else:
        return "Scalene"

    

while 1:
    x, y, z = map(int,input().split())
    if x==0 and y ==0 and z ==0:
        break
    print(division(x,y,z))
    