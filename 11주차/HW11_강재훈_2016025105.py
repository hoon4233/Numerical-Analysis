import cv2
import numpy as np
import pandas as pd
import os

curDir = os.getcwd()
files = os.listdir(curDir)
files.sort()
tmp_list = [0 for _ in range(3)]
RGBrecorr = pd.DataFrame([tmp_list,tmp_list,tmp_list], columns=["R","G","B"], index=["R","G","B"])
YUVrecorr = pd.DataFrame([tmp_list,tmp_list,tmp_list], columns=["Y","U","V"], index=["Y","U","V"])

f = open("result.txt",'a')
for name in files :
    if( name.endswith(".jpeg") ) :
        srcImg = cv2.imread(name, cv2.IMREAD_COLOR)
        yuvImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2YUV)
        
        (B,G,R) = cv2.split(srcImg)
        R = list(np.ravel(R))
        G = list(np.ravel(G))
        B = list(np.ravel(B))

        (Y,U,V) = cv2.split(yuvImg)
        Y = list(np.ravel(Y))
        U = list(np.ravel(U))
        V = list(np.ravel(V))

        df = pd.DataFrame( {"R":R, "G":G, "B":B  } )
        corr = df.corr(method='pearson')
        RGBrecorr += corr.abs()
        f.write(name+"\n")
        f.write(str(corr)+"\n")
        print(name)
        print(corr)
        

        df = pd.DataFrame( {"Y":Y, "U":U, "V":V  } )
        corr = df.corr(method='pearson')
        YUVrecorr += corr.abs()
        f.write(str(corr))
        f.write("\n\n")
        print(corr)
        print("\n\n")

f.write("RGBrecorr\n")
f.write(str(RGBrecorr))
f.write("\n\nYUVrecorr\n")
f.write(str(YUVrecorr))
f.close()

print("RGBrecorr")
print(RGBrecorr)
print("\n\nYUVrecorr")
print(YUVrecorr)

        


