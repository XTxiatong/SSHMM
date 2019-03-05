"""
Created on Tue Jun 12 20:20:29 2018

@author: 111

输入边界的文本文件
存为json文件，是选定范围的边界以及每个街区的中心
"""

import scipy.io as sio    
import matplotlib.pyplot as plt    
import numpy as np    
import json
#2311 街道总数目
R = 675
region = np.empty(R, dtype=object)
center = np.zeros([R,2])
file = 'beijing-lonlat-less3_tencent.txt'
city = 'beijing'

linenumber=0
#计算每个街区的中心
with open(file) as f:
	for line in f:
		temp = line.rstrip('/n').split(' ')[:-1]
		boundary = np.zeros([len(temp)/2,2],dtype=float)
		for i in range (0,len(temp),2):
			boundary[int(i/2)][0]=float(temp[i])
			boundary[int(i/2)][1]=float(temp[i+1])
		region[linenumber] = boundary
		center[linenumber][0] = np.mean(boundary[:,0])
		center[linenumber][1] = np.mean(boundary[:,1])
		print linenumber
		linenumber = linenumber + 1


#筛选出四环以内，可以用框选出一部分，中心在某个经纬度范围内
downtown=[]
for i in range (R):
	#if(center[i][0]>116.20 and center[i][0]<116.58 and center[i][1]>39.75 and center[i][1]<40.07):
   if(1):
		downtown.append(i)

region_id = []
center_downtown = []
downtown_beijing = []
linenumber=0
downtownnumber=0
with open(file) as f:
	for line in f:
		temp = line.strip().split(' ')
		boundary = np.zeros([len(temp)/2,2],dtype=float)
		for i in range (0,len(temp),2):
			boundary[int(i/2)][0]=float(temp[i])
			boundary[int(i/2)][1]=float(temp[i+1])
		if(linenumber in downtown):
			downtown_beijing.append(boundary.tolist())
			center_downtown.append([center[linenumber][0],center[linenumber][1]])
			region_id.append(downtownnumber)
			downtownnumber = downtownnumber + 1
		print linenumber
		linenumber = linenumber + 1
        
#区域id,边界和中心
region_beijing = {"region_id":region_id,"center":center_downtown, "boundary":downtown_beijing}
json.dump(region_beijing, open("region_"+city+"_tencent.json","w"))


#可视化每个区域的中心
plt.figure(figsize=(9,6))
for i in range(len(center_downtown)):
	tmp = center_downtown[i]
	plt.plot(tmp[0],tmp[1],'*')
plt.show()