# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 20:20:29 2018

@author: 111

把边界变成二级编码，目的是加快后一步，但一般是一级码为0
"""
import scipy.io as sio     
import numpy as np    
import json
import math
import matplotlib.pyplot as plt

#北京四环内
city = 'beijing'
region_beijing = json.load(open("region_" + city + "_tencent.json"))
region_id=region_beijing["region_id"]
center_downtown=region_beijing["center"]
downtown_beijing=region_beijing["boundary"]

grid_region = [] # the center of each grid, the fisrt level

#分两级加快索引， 如果街区不是太多，此步可跳过， 则grid_region只有一个点，是选出来的中心，
#最后结果的编码都是0-r, 0是一级编码，r是二级编码
d1at = 0.4
dlog = 0.4
grid_id = 0
for lat in np.arange(116.20,116.58,d1at):
    for log in np.arange(39.75,40.07,dlog):
        grid_region.append([lat+0.5*d1at,log+0.5*dlog])
        
plt.figure(figsize=(8,6))
for i in range(len(grid_region)):
    tmp = grid_region[i]
    plt.plot(tmp[0],tmp[1],'*')
plt.show()

plt.figure(figsize=(8,6))
for i in range(len(center_downtown)):
    tmp = center_downtown[i]
    plt.plot(tmp[0],tmp[1],'*')
plt.show()
       
        
def get_grid_id(point, grids):
    """finding the nearest square id by hierarchical search"""
    dis_cents = 100 #非常大，一开始能进循环
    gc_id = 0
    for i, gc in enumerate(grids):
        dis = math.sqrt((float(point[0]) - float(gc[0])) ** 2 + (float(point[1]) - float(gc[1])) ** 2)
        if dis < dis_cents:
            dis_cents = dis
            gc_id = i
    return gc_id

grid_boundary = {}

for index, center in enumerate(center_downtown):
    grid_id = get_grid_id(center,grid_region)
    if (grid_id not in grid_boundary.keys()):
        grid_boundary[grid_id]={}
        grid_boundary[grid_id][index]=downtown_beijing[index]
    else:
        grid_boundary[grid_id][index]=downtown_beijing[index]   
#    #print index

#grid_region是一级网格的地理中心，用来判断属于哪个一级网格，grid_boundary的键值是一级网格id,value是的key是二级id和边界
region = {'grid_region':grid_region,'grid_boundary':grid_boundary,'center_downtown':center_downtown}
json.dump(region, open("region_" + city +"_tencent2.json","w"))   
    

    