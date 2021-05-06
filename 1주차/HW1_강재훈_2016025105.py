def y(x):
    ret = 5*(x**4) - 22.4*(x**3) + 15.85272*(x**2) + 24.161472*(x) - 23.4824832
    return ret
def yy(x):
    ret = 5*4*(x**3) - 22.4*3*(x**2) + 15.85272*2*(x) + 24.161472
    return ret

def bisection(left, right, t):
    if y(left) * y(right) > 0 :
        pass
    else :
        while (right-left)/2.0 > t :
            mid = (right+left)/2.0
            if y(mid) == 0 :
                return mid
            elif y(left) * y(mid) < 0 :
                right = mid
            else :
                left = mid
        return mid

def newton_raphson(point,t):
    while abs(y(point)/yy(point)) >= t :
        point = point - (y(point)/yy(point))
    return point

t = 0.00001
print("Bisection")
for i  in range(-4,4):
    print("i : "+str(i)+", root :",bisection(i,i+1,t))
print()
print("Newton_raphson")
for i in range(-4,5,2):
    print("i : "+str(i)+", root :",newton_raphson(i,t))

# print(bisection(-2,0,t))
# print(bisection(0,2,t))
# print(bisection(2,4,t))
# print()
# print("Newton_rhapson")
# print(newton_rhapson(-2,t))
# print(newton_rhapson(0,t))
# print(newton_rhapson(4,t))