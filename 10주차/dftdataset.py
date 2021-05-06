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
    # print(np.abs(block))
    block = np.fft.ifftshift(block)
    return block

def calMean(filename):
    global curDir, blocks, block_size

    M = np.zeros((block_size,block_size))
    M = M.astype(np.complex128)

    for i in range(blocks):
        for j in range(blocks):
            tmp = fftBlock(filename,i*block_size,(i+1)*block_size ,j*block_size,(j+1)*block_size)
            M += tmp

    M = M / (blocks*blocks)

    return M

def meanifft(filename):
    M = calMean(filename)
    img_back = np.abs( np.fft.ifft2(M) )
    
    plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
    plt.title('mean Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def cmpself(filename):
    global curDir, blocks,block_size

    M = calMean(filename)
    result = np.zeros((block_size,block_size))
    result = M.astype(np.complex128)

    for i in range(blocks):
        for j in range(blocks):
            tmp = fftBlock(filename,i*block_size,(i+1)*block_size ,j*block_size,(j+1)*block_size)
            result += (tmp-M) * (tmp-M)

    result = result / (blocks*blocks)

    # f = open(curDir+"/"+"dftmean.txt",'a')
    # f.write(filename+"\n")
    # f.write(str(result.sum())+"\n")
    # f.close()

    return result

def cmpother(filename, filename2):
    global curDir, blocks,block_size

    M = calMean(filename)
    result = np.zeros((block_size,block_size))
    result = M.astype(np.complex128)

    for i in range(blocks):
        for j in range(blocks):
            tmp = fftBlock(filename2,i*block_size,(i+1)*block_size ,j*block_size,(j+1)*block_size)
            result += (tmp-M) * (tmp-M)

    result = result / (blocks*blocks)
    # S = result.sum().real - cmpself(filename).sum().real
    S = result.sum().real
    
    f = open(curDir+"/"+"result.txt",'a')
    f.write(filename+" vs "+filename2+"\n")
    f.write(str(S)+"\n"+"\n")
    f.close()

    return result



file_num = 20
for i in range(file_num):
    f = open(curDir+"/"+"result.txt",'a')
    f.write("\n")
    f.close()
    for j in range(file_num):
        cmpother(str(i)+".jpg", str(j)+".jpg")
    
    f = open(curDir+"/"+"dftmean.txt",'a')
    f.write("-------------------------------------------")
    f.write("\n")
    f.close()

# file_num = 20
# for i in range(file_num): 
#     print(cmpself(str(i)+".jpg").sum().real)



def fftBlock( filename, start_x, end_x, start_y, end_y ):
def calMean(filename):
def meanifft(filename):
def cmpself(filename):    
def cmpother(filename, filename2):
def dominantC(filename):