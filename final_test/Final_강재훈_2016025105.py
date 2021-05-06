import cv2
import numpy as np

train_sample_num = 300
test_sample_num = 100

x_1, y_1, z_1 = np.random.normal(0, 1, 400), np.random.normal(0, 1, 400), np.random.normal(0, 1, 400)
x_2, y_2, z_2 = np.random.normal(1, 1**0.5, 400), np.random.normal(-1, 1**0.5, 400), np.random.normal(0, 3**0.5, 400)
x_3, y_3, z_3 = np.random.normal(0, 3**0.5, 400), np.random.normal(-1, 1**0.5, 400), np.random.normal(2, 1**0.5, 400)
x_4, y_4, z_4 = np.random.normal(-2, 1**0.5, 400), np.random.normal(0, 3**0.5, 400), np.random.normal(2, 1**0.5, 400)
x_5, y_5, z_5 = np.random.normal(0, 3**0.5, 400), np.random.normal(1, 1**0.5, 400), np.random.normal(2, 1**0.5, 400)

x_6, y_6, z_6 = np.random.normal(1, 1**0.5, 100), np.random.normal(1, 1**0.5, 100), np.random.normal(-1, 3**0.5, 100)

max_dis = [-1 for _ in range(5)]
min_dis = [1000 for _ in range(5)]


#make train_set
train_point = []
for i in range(train_sample_num):
    train_point.append( [x_1[i], y_1[i], z_1[i]] )
    train_point.append( [x_2[i], y_2[i], z_2[i]] )
    train_point.append( [x_3[i], y_3[i], z_3[i]] )
    train_point.append( [x_4[i], y_4[i], z_4[i]] )
    train_point.append( [x_5[i], y_5[i], z_5[i]] )

#k-cluster
ret,label,center=cv2.kmeans(np.float32(train_point), 5, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1), 10, cv2.KMEANS_RANDOM_CENTERS)

print("center : ")
print(center)
# print("label : ")
label = label.flatten()
# print(label)
# for i in range(15):
#     print(label[i])

#cal max_dis for each cluster
for i in range(train_sample_num*5) :
    max_dis[ label[i] ] = max(max_dis[ label[i] ], np.linalg.norm(center[ label[i] ]-train_point[i])  )
    min_dis[ label[i] ] = min(min_dis[ label[i] ], np.linalg.norm(center[ label[i] ]-train_point[i])  )

print("\nMax_dis & Min_dis :")
print(max_dis, min_dis)


#select cluster for each cluster
select_cluster =  [ [0, 0, 0, 0, 0] for _ in range(5) ]  #각각의 가우시안이 갖는 cluster 찾아주기
tmp = 0
for i in label:
    if tmp == 5 :
        tmp = 0
    select_cluster[tmp][i] += 1
    tmp += 1

print("\nSelect_cluster : ")
print(select_cluster)
for i in range(len(select_cluster)) :
    print("%d distribution == %d cluster" %(i+1, select_cluster[i].index(max(select_cluster[i]))+1) )
print("\n")
        

#test
for i in range(train_sample_num,train_sample_num+test_sample_num,1):
    #1 distribution
    if ( np.linalg.norm( [ x_1[i], y_1[i], z_1[i] ] - center[ select_cluster[0].index(max(select_cluster[0])) ] ) < max_dis[0] ) :
        print("First distribution's point (index : %d) is in original cluster" %(i))
    else :
        tmp_min = 123456789
        tmp_idx = -1
        for j in range(5):
            tmp_dis = np.linalg.norm( [ x_1[i], y_1[i], z_1[i] ] - center[j] )
            if tmp_dis <= max_dis[j] :
                if tmp_min > tmp_dis :
                    tmp_min = tmp_dis
                    tmp_idx = j

        if tmp_min != 123456789 :
            print("First distribution's point (index : %d) is not in original cluster.(Instead in %d cluster)" %(i,tmp_idx+1))
        else :
            print("First distribution's point (index : %d) is not in any cluster" %(i))

    #2 distribution
    if ( np.linalg.norm( [ x_2[i], y_2[i], z_2[i] ] - center[ select_cluster[1].index(max(select_cluster[1])) ] ) < max_dis[1] ) :
        print("Second distribution's point (index : %d) is in original cluster" %(i))
    else :
        tmp_min = 123456789
        tmp_idx = -1
        for j in range(5):
            tmp_dis = np.linalg.norm( [ x_2[i], y_2[i], z_2[i] ] - center[j] )
            if tmp_dis <= max_dis[j] :
                if tmp_min > tmp_dis :
                    tmp_min = tmp_dis
                    tmp_idx = j

        if tmp_min != 123456789 :
            print("Second distribution's point (index : %d) is not in original cluster.(Instead in %d cluster)" %(i,tmp_idx+1))
        else :
            print("Second distribution's point (index : %d) is not in any cluster" %(i))


    #3 distribution
    if ( np.linalg.norm( [ x_3[i], y_3[i], z_3[i] ] - center[ select_cluster[2].index(max(select_cluster[2])) ] ) < max_dis[2] ) :
        print("Third distribution's point (index : %d) is in original cluster" %(i))
    else :
        tmp_min = 123456789
        tmp_idx = -1
        for j in range(5):
            tmp_dis = np.linalg.norm( [ x_3[i], y_3[i], z_3[i] ] - center[j] )
            if tmp_dis <= max_dis[j] :
                if tmp_min > tmp_dis :
                    tmp_min = tmp_dis
                    tmp_idx = j

        if tmp_min != 123456789 :
            print("Third distribution's point (index : %d) is not in original cluster.(Instead in %d cluster)" %(i,tmp_idx+1))
        else :
            print("Third distribution's point (index : %d) is not in any cluster" %(i))

    
    #4 distribution
    if ( np.linalg.norm( [ x_4[i], y_4[i], z_4[i] ] - center[ select_cluster[3].index(max(select_cluster[3])) ] ) < max_dis[3] ) :
        print("Forth distribution's point (index : %d) is in original cluster" %(i))
    else :
        tmp_min = 123456789
        tmp_idx = -1
        for j in range(5):
            tmp_dis = np.linalg.norm( [ x_4[i], y_4[i], z_4[i] ] - center[j] )
            if tmp_dis <= max_dis[j] :
                if tmp_min > tmp_dis :
                    tmp_min = tmp_dis
                    tmp_idx = j

        if tmp_min != 123456789 :
            print("Forth distribution's point (index : %d) is not in original cluster.(Instead in %d cluster)" %(i,tmp_idx+1))
        else :
            print("Forth distribution's point (index : %d) is not in any cluster" %(i))

    #5 distribution
    if ( np.linalg.norm( [ x_5[i], y_5[i], z_5[i] ] - center[ select_cluster[4].index(max(select_cluster[4])) ] ) < max_dis[4] ) :
        print("Fifth distribution's point (index : %d) is in original cluster" %(i))
    else :
        tmp_min = 123456789
        tmp_idx = -1
        for j in range(5):
            tmp_dis = np.linalg.norm( [ x_5[i], y_5[i], z_5[i] ] - center[j] )
            if tmp_dis <= max_dis[j] :
                if tmp_min > tmp_dis :
                    tmp_min = tmp_dis
                    tmp_idx = j

        if tmp_min != 123456789 :
            print("Fifth distribution's point (index : %d) is not in original cluster.(Instead in %d cluster)"%(i,tmp_idx+1))
        else :
            print("Fifth distribution's point (index : %d) is not in any cluster" %(i))

    print("\n")


for i in range(test_sample_num) :
    tmp_min = 123456789
    tmp_idx = -1
    for j in range(5):
        tmp_dis = np.linalg.norm( [ x_6[i], y_6[i], z_6[i] ] - center[j] )
        if tmp_dis <= max_dis[j] :
            if tmp_min > tmp_dis :
                tmp_min = tmp_dis
                tmp_idx = j

    if tmp_min != 123456789 :
            print("Sixth distribution's point (index : %d) is in %d cluster" %(i,tmp_idx+1))
    else :
        print("Sixth distribution's point (index : %d) is not in any cluster" %(i))