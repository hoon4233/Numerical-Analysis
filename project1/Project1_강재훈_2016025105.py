import numpy as np
import cv2
import os

curDir = os.getcwd()
srcDir = "/newdataset"
dstDir = "/grayset"
testDir = "/testset"
recoveryDir = "/recoveryset"
NUMBER_OF_DATA = 1285
WIDTH = 32
HEIGHT = 32
MEAN = 0
COMPRESSION = 40
A = np.ndarray( shape = (NUMBER_OF_DATA, WIDTH * HEIGHT), dtype = np.float64 )
ori_A = np.ndarray( shape = (NUMBER_OF_DATA, WIDTH * HEIGHT), dtype = np.float64 )
eigenFace = np.ndarray( shape = (COMPRESSION, 1024), dtype = np.float64 )

everyCoeffi = [ ]
meResult = [[] for _ in range(10) ]
manyResult = [[] for _ in range(10) ] 
manyResult2 = [[] for _ in range(10) ] 

def makeDataSet():
    global curDir, srcDir, dstDir

    files = os.listdir(curDir+srcDir)
    files.sort()
    name = 0
    for i in range(len(files)):
        if( files[i].endswith(".pgm") ) :
            srcImg = cv2.imread(curDir+'/'+srcDir+'/'+files[i], cv2.IMREAD_COLOR)
            dstImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
            finImg = cv2.resize(dstImg, dsize=(32,32), interpolation=cv2.INTER_AREA)
            cv2.imwrite(curDir+'/'+dstDir+'/'+ str(name) +".pgm", finImg)
            name += 1
    

# def makeDataSet2():
#     global curDir, srcDir, dstDir
#     subDir = os.listdir(curDir+srcDir)
#     subDir.sort() #1부터 len(subdir)-1까지 값이 들어있다.
#     name = 0
#     for i in range(1,len(subDir),1):
#         nowDir = curDir + srcDir + '/' + subDir[i]
#         nowfile = os.listdir(nowDir)
#         for j in range(0,len(nowfile),1):
#             if( nowfile[j].endswith(".jpg") ) :
#                 srcImg = cv2.imread(nowDir+'/'+nowfile[j], cv2.IMREAD_COLOR)
#                 dstImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
#                 finImg = cv2.resize(dstImg, dsize=(32,32), interpolation=cv2.INTER_AREA)
#                 cv2.imwrite(curDir+'/'+dstDir+'/'+ str(name) +".jpg", finImg)
#                 name += 1

def makeA():
    global NUMBER_OF_DATA, WIDHT, HEIGHT, MEAN, A
    ori_A = np.ndarray( shape = (NUMBER_OF_DATA, WIDTH * HEIGHT), dtype = np.float64 )

    files = os.listdir(curDir+dstDir)
    files.sort()
    
    for i in range(len(files)):
        if( files[i].endswith(".pgm") ) :
            img = cv2.imread( curDir + '/' +dstDir + '/' + files[i], cv2.IMREAD_GRAYSCALE )
            ori_A[i] = img.flatten().astype('float64')
        
    MEAN = np.mean(ori_A, axis = 0)
    for i in range(NUMBER_OF_DATA):
        A[i] = ori_A[i] - MEAN

def makeEigenface():
    global A, eigenFace, COMPRESSION
    ATA = np.dot(A.T,A)
    U, s, V = np.linalg.svd(ATA, full_matrices = True)
    pyS = s.tolist() #s를 파이썬 리스트로 변경
    sortS = sorted(pyS, reverse=True) #eigen value가 큰 걸 찾기 위해 정렬
    for i in range(COMPRESSION):
        eigenFace[i] = V[ pyS.index(sortS[i]) ]

def calCoeffi(ori_face):
    global eigenFace, COMPRESSION
    cList = []
    for i in range(COMPRESSION):
        cList.append( np.dot(ori_face, eigenFace[i]) )
    
    return cList

def makeOriImage(coeffi, filename) :
    global eigenFace, MEAN, curDir, recovery
    B = np.ndarray( shape = (1024), dtype = np.float64 )

    for i in range(len(eigenFace)):
        B += np.dot(coeffi[i], eigenFace[i])
        print("Coefficients : ",np.array(coeffi[i]))
        print("EigenFace : ",eigenFace[i])
        print()
    B += MEAN
    B = B.reshape(32,32).astype(np.int64)
    cv2.imwrite(curDir+'/'+recoveryDir+'/'+ filename, B)
    print("IN makeOri : ", filename)
    print(B.astype(np.int64))

def onePersonManyImg(name) : #사람 한명에 대해 여러장의 사진을 이용하여 coefficient들의 값 비교
    global curDir, testDir, MEAN, everyCoeffi
    nowDir = curDir + testDir +'/' + name
    nowfile = os.listdir(nowDir)
    coeffiS = []
    for i in range(0, len(nowfile),1 ) : 
        if( nowfile[i].endswith(".pgm") ) :
            srcImg = cv2.imread(nowDir+'/'+nowfile[i], cv2.IMREAD_COLOR)
            dstImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
            finImg = cv2.resize(dstImg, dsize=(32,32), interpolation=cv2.INTER_AREA)
            print("IN onePerson : ", nowfile[i])
            print(finImg)
            

            img = finImg.flatten().astype('float64')
            img -= MEAN


            coeffi = calCoeffi(img)
            makeOriImage(coeffi, nowfile[i])

            coeffiS.append( coeffi )

    everyCoeffi.append(coeffiS)
    
        
def cmpMe():
    global everyCoeffi, meResult
    everyCoeffi = np.array(everyCoeffi)
    pNum = 10
    iNum = 5
    for i in range(pNum):
        for j in range(iNum):
            for k in range(4,j,-1):
                tmp = everyCoeffi[i][j] - everyCoeffi[i][k]
                meResult[i].append(np.linalg.norm(tmp))

    for i in range(10):
        print("Person number", i+1 )
        print(meResult[i])
        print("MEAN : ",np.mean(meResult[i]))
        print()



def cmpPerson() : #다른 사람이면 coefficient가 어떻게 다른가 비교
    global everyCoeffi, manyReult, manyResult2
    everyCoeffi = np.array(everyCoeffi)
    pNum = 10
    iNum = 5
    tmp_k, tmp_l = 0,0
    for i in range(pNum):
        for j in range(i+1,pNum,1):
            tmpN = 10e9
            for k in range(iNum):
                for l in range(iNum):
                    tmpL = everyCoeffi[i][k] - everyCoeffi[j][l]
                    a = np.linalg.norm(tmpL)
                    if(tmpN > a):
                        tmp_k, tmp_l = k,l
                        tmpN = a
            manyResult[i].append([tmp_k,tmp_l,tmpN])

    tmp_k, tmp_l = 0,0
    for i in range(pNum):
        for j in range(i+1,pNum,1):
            tmpN = -1
            for k in range(iNum):
                for l in range(iNum):
                    tmpL = everyCoeffi[i][k] - everyCoeffi[j][l]
                    a = np.linalg.norm(tmpL)
                    if(tmpN < a):
                        tmp_k, tmp_l = k,l
                        tmpN = a
            manyResult2[i].append([tmp_k,tmp_l,tmpN])
    
    print("Min result")
    for i in range(pNum):
        print("Person number", i+1 )
        print(manyResult[i])
        print()

    print("Max result")
    for i in range(pNum):
        print("Person number", i+1 )
        print(manyResult2[i])
        print()

    


# makeDataSet()
makeA()
makeEigenface()
onePersonManyImg("Abdullah_Gul")
onePersonManyImg("Adrien_Brody")
onePersonManyImg("Ahmed_Chalabi")
onePersonManyImg("Al_Gore")
onePersonManyImg("Al_Sharpton")
onePersonManyImg("Alastair_Campbell")
onePersonManyImg("Albert_Costa")
onePersonManyImg("Alejandro_Toledo")
onePersonManyImg("Ana_Guevara")
onePersonManyImg("Ana_Palacio")
# cmpMe()
cmpPerson()
