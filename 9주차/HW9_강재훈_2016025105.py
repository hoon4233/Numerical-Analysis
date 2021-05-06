#test case1
import cv2
import numpy as np

N = 16

def aa(u,v):
    result = 1

    if u == 0 and v == 0: 
        result = 1/16

    elif (u == 0 and v > 0) :
        result = 2/16
    elif (u > 0 and v == 0) :
        result = 2/16

    else: 
        result = 4/16

    return result


def dct(block):
    c = np.zeros((N,N),np.float32)
    
    for u in range(N):
        for v in range(N):
            tmp = 0
            for x in range(N):
                for y in range(N):
                    tmp += block[x][y] * np.cos( ( np.pi*(2*x+1)*u ) / (2*N) ) * np.cos( ( np.pi*(2*y+1)*v ) / (2*N) )

            tmp = aa(u,v)*tmp
            c[u][v] = tmp

    rank = np.ravel(np.abs(c))
    rank = np.sort(rank)[::-1]
    idx = []

    for i in range(N):
        if len(idx) == 16 :
                    break
        for j in range(N):
            if len(idx) == 16 :
                    break
            for k in range(N):
                if len(idx) == 16 :
                    break
                if c[j][k] == rank[i] :
                    idx.append([j,k,rank[i]])
                    
    c = np.zeros((N,N),np.float32)
    for x,y,v in idx :
        c[x][y] = v

    return c

def idct(c):
    f = np.zeros((N,N),np.float32)
    for x in range(N):
        for y in range(N):
            tmp = 0
            for u in range(N):
                for v in range(N):
                    tmp += aa(u,v) * c[u][v] * np.cos( ( np.pi*(2*x+1)*u ) / (2*N) ) * np.cos( ( np.pi*(2*y+1)*v ) / (2*N)  )
            f[x][y] = tmp

    return f




file = "test1.jpg"
img = cv2.imread(file,cv2.IMREAD_COLOR)

h, w = np.array(img.shape[:2])
(b, g, r) = cv2.split(img)

for colors in range(3):
    color = np.zeros((h,w),np.float32)
    if colors == 0 :
        color[:h,:w] = r
    elif colors == 1 :
        color[:h,:w] = g
    else :
        color[:h,:w] = b

    print(color)
    print(color.shape)

    for row in range(h//N):
        for col in range(w//N):
            currentblock = dct(color[row*N:(row+1)*N, col*N:(col+1)*N])
            currentblock = idct(currentblock)
            
            for i in range(N):
                for j in range(N):
                    tmp = currentblock[i][j]
                    if tmp < 0 :
                        tmp = 0
                    if tmp > 255 :
                        tmp = 255

                    if colors == 0 :
                        r[row*N+i][col*N+j] = tmp
                    elif colors == 1 :
                        g[row*N+i][col*N+j] = tmp
                    else :
                        b[row*N+i][col*N+j] = tmp

    print("color : ",colors, " DCT & IDCT is finish")

result = cv2.merge([b,g,r])
cv2.imshow("result", result)
cv2.imwrite("result1.jpg",result)
cv2.waitKey(0)



#test case2
# import cv2
# import numpy as np

# N = 16

# def aa(u,v):
#     result = 1

#     if u == 0 and v == 0: 
#         result = 1/16

#     elif (u == 0 and v > 0) :
#         result = 2/16
#     elif (u > 0 and v == 0) :
#         result = 2/16

#     else: 
#         result = 4/16

#     return result


# def dct(block):
#     c = np.zeros((N,N),np.float32)
    
#     for u in range(N):
#         for v in range(N):
#             tmp = 0
#             for x in range(N):
#                 for y in range(N):
#                     tmp += block[x][y] * np.cos( ( np.pi*(2*x+1)*u ) / (2*N) ) * np.cos( ( np.pi*(2*y+1)*v ) / (2*N) )

#             tmp = aa(u,v)*tmp
#             c[u][v] = tmp

#     rank = np.ravel(np.abs(c))
#     rank = np.sort(rank)[::-1]
#     pin = rank[N]

#     check = 0
#     for i in range(N):
#         for j in range(N):
#             if check == N :
#                 c[i][j] = 0
#                 continue

#             if abs(c[i][j]) > pin :
#                 check += 1
#             if abs(c[i][j]) <= pin :
#                 c[i][j] = 0
                    
#     return c

# def idct(c):
#     f = np.zeros((N,N),np.float32)
#     for x in range(N):
#         for y in range(N):
#             tmp = 0
#             for u in range(N):
#                 for v in range(N):
#                     tmp += aa(u,v) * c[u][v] * np.cos( ( np.pi*(2*x+1)*u ) / (2*N) ) * np.cos( ( np.pi*(2*y+1)*v ) / (2*N)  )
#             f[x][y] = tmp

#     return f




# file = "test2.jpg"
# img = cv2.imread(file,cv2.IMREAD_COLOR)

# h, w = np.array(img.shape[:2])
# (b, g, r) = cv2.split(img)

# for colors in range(3):
#     color = np.zeros((h,w),np.float32)
#     if colors == 0 :
#         color[:h,:w] = r
#     elif colors == 1 :
#         color[:h,:w] = g
#     else :
#         color[:h,:w] = b

#     print(color)
#     print(color.shape)

#     for row in range(h//N):
#         for col in range(w//N):
#             currentblock = dct(color[row*N:(row+1)*N, col*N:(col+1)*N])
#             currentblock = idct(currentblock)
            
#             for i in range(N):
#                 for j in range(N):
#                     tmp = currentblock[i][j]
#                     if tmp < 0 :
#                         tmp = 0
#                     if tmp > 255 :
#                         tmp = 255

#                     if colors == 0 :
#                         r[row*N+i][col*N+j] = tmp
#                     elif colors == 1 :
#                         g[row*N+i][col*N+j] = tmp
#                     else :
#                         b[row*N+i][col*N+j] = tmp

#     print("color : ",colors, " DCT & IDCT is finish")

# result = cv2.merge([b,g,r])
# cv2.imshow("result", result)
# cv2.imwrite("result2.jpg",result)
# cv2.waitKey(0)





#test case3
# import cv2
# import numpy as np

# N = 16

# def aa(u,v):
#     result = 1

#     if u == 0 and v == 0: 
#         result = 1/16

#     elif (u == 0 and v > 0) :
#         result = 2/16
#     elif (u > 0 and v == 0) :
#         result = 2/16

#     else: 
#         result = 4/16

#     return result


# def dct(block):
#     c = np.zeros((N,N),np.float32)
    
#     for u in range(N):
#         for v in range(N):
#             tmp = 0
#             for x in range(N):
#                 for y in range(N):
#                     tmp += block[x][y] * np.cos( ( np.pi*(2*x+1)*u ) / (2*N) ) * np.cos( ( np.pi*(2*y+1)*v ) / (2*N) )

#             tmp = aa(u,v)*tmp
#             c[u][v] = tmp

#     rank = np.ravel(np.abs(c))
#     rank = np.sort(rank)[::-1]
#     pin = rank[N]

#     check = 0
#     for i in range(N):
#         for j in range(N):
#             if check == N :
#                 c[i][j] = 0
#                 continue

#             if abs(c[i][j]) > pin :
#                 check += 1
#             if abs(c[i][j]) <= pin :
#                 c[i][j] = 0
                    
#     return c

# def idct(c):
#     f = np.zeros((N,N),np.float32)
#     for x in range(N):
#         for y in range(N):
#             tmp = 0
#             for u in range(N):
#                 for v in range(N):
#                     tmp += aa(u,v) * c[u][v] * np.cos( ( np.pi*(2*x+1)*u ) / (2*N) ) * np.cos( ( np.pi*(2*y+1)*v ) / (2*N)  )
#             f[x][y] = tmp

#     return f




# file = "test3.jpg"
# img = cv2.imread(file,cv2.IMREAD_COLOR)

# h, w = np.array(img.shape[:2])
# (b, g, r) = cv2.split(img)

# for colors in range(3):
#     color = np.zeros((h,w),np.float32)
#     if colors == 0 :
#         color[:h,:w] = r
#     elif colors == 1 :
#         color[:h,:w] = g
#     else :
#         color[:h,:w] = b

#     print(color)
#     print(color.shape)

#     for row in range(h//N):
#         for col in range(w//N):
#             currentblock = dct(color[row*N:(row+1)*N, col*N:(col+1)*N])
#             currentblock = idct(currentblock)
            
#             for i in range(N):
#                 for j in range(N):
#                     tmp = currentblock[i][j]
#                     if tmp < 0 :
#                         tmp = 0
#                     if tmp > 255 :
#                         tmp = 255

#                     if colors == 0 :
#                         r[row*N+i][col*N+j] = tmp
#                     elif colors == 1 :
#                         g[row*N+i][col*N+j] = tmp
#                     else :
#                         b[row*N+i][col*N+j] = tmp

#     print("color : ",colors, " DCT & IDCT is finish")

# result = cv2.merge([b,g,r])
# cv2.imshow("result", result)
# cv2.imwrite("result3.jpg",result)
# cv2.waitKey(0)
