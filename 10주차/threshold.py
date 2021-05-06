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
    alpha = 100
    a = [[alpha for i in range(block_size)] for j in range(block_size)]
    a = np.array(a)
    M = calMean(filename) + a
    img_back = np.fft.ifft2(M) 
    img_back = np.abs( np.fft.ifft2(M) )
    
    plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
    plt.title('mean Image'), plt.xticks([]), plt.yticks([])
    plt.show()

meanifft("1.jpg")
alpha = 100
a = [[alpha for i in range(block_size)] for j in range(block_size)]
a = np.array(a)
b = [[0 for i in range(block_size)] for j in range(block_size)]
b= np.array(b)
for i in range(16):
    b += a*a
b = b/16
print(b.sum().real)