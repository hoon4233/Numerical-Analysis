import numpy as np

x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
noise_y =  np.array([-9.385465225173377, -12.414363749437356, -6.609567429268855, -5.639109976112357, -5.861720754283326, -1.5525548588979334, -0.7271662969030217, 5.496927140772376, 3.7354177602376195, 6.968831157394976, 9.603880931397494, 10.77116152245486] )
min_points = [[-3, -6.609567429268855], [-2, -5.639109976112357], [0, -1.5525548588979334], [4, 6.968831157394976], [5, 9.603880931397494], [6, 10.77116152245486]]
min_x = []
min_y = []
for a,b in min_points :
    min_x.append(a)
    min_y.append(b)

min_x = np.array(min_x)
min_y = np.array(min_y)

A = np.vstack([x, np.ones(len(x))]).T
A2 = np.vstack([min_x, np.ones(len(min_x))]).T

m1, c1 = np.linalg.lstsq(A, noise_y, rcond=None)[0]
m2, c2 = np.linalg.lstsq(A2, min_y, rcond=None)[0]

print("Using 12 sample y = %fx +(%f)" %(m1,c1))
print("Using 6 sample y = %fx +(%f)" %(m2,c2))





