import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

curDir = os.getcwd()
blocks = 4
block_size = 64

def fftBlock( filename, start_x, end_x, start_y, end_y ):
    global curDir
    
    srcImg = cv2.imread(curDir+'/'+filename, cv2.COLOR_BGR2GRAY)
    tarImg = srcImg[start_x:end_x, start_y:end_y]
    block = np.fft.fft2(tarImg)
    block = np.fft.fftshift(block)
    block[block_size//2, block_size//2] = 0
    block = np.fft.ifftshift(block)
    return block


def dominantC(filename):
    global curDir, blocks,block_size

    f = open(curDir+"/"+"dominantC.txt",'a')
    f.write(filename+"\n")
    for i in range(blocks):
        for j in range(blocks):
            f.write("\n"+ "row : " + str(i) + " col : " +str(j) + "\n")
            tmp = fftBlock(filename,i*block_size,(i+1)*block_size ,j*block_size,(j+1)*block_size)
            print(tmp.shape)
            tmp = np.ravel(np.abs(tmp))
            tmp = list(tmp)
            tmp2 = []
            for k in range(15):
                l = tmp.index(max(tmp))
                r,c = l//block_size, l%block_size
                tmp2.append([r,c,tmp[l]])
                tmp[l] = -1

            count = 1
            for a,b,c in tmp2:
                f.write( "dominant"+ str(count) +" : " + "[" + str(a) + "]" + "[" + str(b) + "] " + str(c) + "\n")
                count += 1

    f.close()





file_num = 20
for i in range(file_num): 
    f = open(curDir+"/"+"dominantC.txt",'a')
    f.write("------------------------------------"+"\n")
    f.close()
    dominantC(str(i)+".jpg")
