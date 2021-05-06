import numpy as np
import random
import itertools
def ROW(x) : 
    l = [x**2, x, 1]
    return l
def pseudoInverse(A,b):
    AtA = np.dot( np.transpose(A), A )
    AtA_1At = np.dot( np.linalg.inv(AtA), np.transpose(A) )
    x = np.dot(AtA_1At,b)
    return x

points = [ [-2.9, 35.4], [-2.1, 19.7], [-0.9, 5.7], [1.1, 2.1], [0.1, 1.2], [1.9, 8.7], [3.1, 25.7], [4.0, 41.5] ]
C = list(itertools.combinations(points,6))
idx1, idx2 = 0, 0
l = len(C)-1
while idx1 == idx2 :
    idx1, idx2 = random.randint(0,l), random.randint(0,l)
case1, case2 = C[idx1], C[idx2]
A1, A2, b1, b2 = [], [], [], []

for i in range(6):
    A1.append(ROW(case1[i][0]))
    b1.append(case1[i][1])
    A2.append(ROW(case2[i][0]))
    b2.append(case2[i][1])

A1, A2, b1, b2 = np.array(A1), np.array(A2), np.array(b1), np.array(b2)
result1, result2 = pseudoInverse(A1,b1), pseudoInverse(A2,b2)
print("case1 points :",case1, "& result1 :",result1)
print("case2 points :",case2, "& result2 :",result2)