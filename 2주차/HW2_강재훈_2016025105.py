def y(x):
    ret = 5*(x**4) - 22.4*(x**3) + 15.85272*(x**2) + 24.161472*(x) - 23.4824832
    return ret
def yy(x):
    ret = 5*4*(x**3) - 22.4*3*(x**2) + 15.85272*2*(x) + 24.161472
    return ret
def yyy(x):
    ret = 5*4*3*(x**2) - 22.4*3*2*(x) + 15.85272*2
    return ret
def yy2(x,h):
    ret = ( y(x+h) - y(x) ) / (h)
    return ret
def yyy2(x,h):
    ret = ( y(x+h) - 2*y(x) + y(x-h) ) / (h**2)
    return ret

def first_method(point,t):
    while abs(yy(point)/yyy(point)) >= t :
        point = point - (yy(point)/yyy(point))
    return point

def second_method(point,t,h):
    while abs(yy2(point,h)/yyy2(point,h)) >= t :
        point = point - (yy2(point,h)/yyy2(point,h))
    return point

t = 0.00001
h = 0.0001
print("first_method")
for i in range(-4,4,1):
    print("i : "+str(i)+", root :",first_method(i,t))
print("second_method")
for i in range(-4,4,1):
    print("i : "+str(i)+", root :",second_method(i,t,h))
