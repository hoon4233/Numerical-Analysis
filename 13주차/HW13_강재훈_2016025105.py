import itertools
import numpy as np

def caly(x):
    ret = 2*x-1
    return ret

def leastsquare(points):
    global max_error, min_error
    error = 0
    for i in range(6):
        error += (points[i][1] - caly(points[i][0])) ** 2
    max_error = max(max_error, error)
    min_error = min(min_error, error)

    print(points)
    print(error)
    print("now min_error : ",min_error,"\n")



point_num = 12
x = [i for i in range(-5,7,1)]
noise = np.random.normal(0,2**0.5,12)
real_y = [ caly(i) for i in x ]

noise_y = []
for i in range(point_num):
    noise_y.append(real_y[i] + noise[i])

print("x : ", x)
print("real_y : ", real_y)
print("noise : ", noise)
print("noise_y : ",noise_y,"\n")

noise_point = []
max_error = -1.0
min_error = 100.0

for i in range(point_num):
    a,b = x[i], noise_y[i]
    noise_point.append([a,b])

com_noise = list(itertools.combinations(noise_point,6))

for i in com_noise:
    leastsquare(i)

print("MAX : ", max_error)
print("MIN : ", min_error)


