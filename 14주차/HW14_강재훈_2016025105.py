from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2
import numpy as np

srcImg = cv2.imread("1.png", cv2.IMREAD_COLOR)
conImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2LAB)
reImg = np.reshape(conImg, [-1,3])

bandwidth = estimate_bandwidth(reImg, quantile = 0.03, n_samples = 1000)
mean_sh = MeanShift(bandwidth = bandwidth, bin_seeding = True)
mean_sh.fit(reImg)
labels, centers = mean_sh.labels_, mean_sh.cluster_centers_
center = np.uint8(centers)
res = center[labels.flatten()]
res = res.reshape((srcImg.shape))
cluster_num = len(np.unique(labels))

print("How many clusters(mode) ? : ", cluster_num)

res = cv2.cvtColor(res,cv2.COLOR_LAB2BGR)
cv2.imwrite("mean-1.png",res)



reImg = np.float32(reImg)
K = cluster_num
ret,label,center=cv2.kmeans(reImg, K, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res = res.reshape((srcImg.shape))
res = cv2.cvtColor(res,cv2.COLOR_LAB2BGR)
cv2.imwrite("k-1.png",res)