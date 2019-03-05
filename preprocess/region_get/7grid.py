# -*- coding: utf-8 -*-
"""
Created on Thu Jul 05 14:43:21 2018
@author: 111

经纬度转成莫卡托网格再转成region, 直接由经纬度映射到街区

这个代码很耗时，但是转完以后，在处理checkin,POI,轨迹都会比较容易
"""
import datetime
from math import log, tan, sin, cos, atan, sqrt, exp
import json
import numpy as np

earth_radius = 6378137.0
pi = 3.1415926535897932384626
meter_per_degree = earth_radius * pi / 180.0

def transfer_lnglat_to_mercator(degree_lng, degree_lat):
    meter_lng = degree_lng * meter_per_degree
    meter_lat = log(tan((90.0 + degree_lat) * pi / 360.0)) * earth_radius
    return meter_lng, meter_lat

def transfer_mercator_to_lnglat(meter_lng, meter_lat):
    degree_lng = meter_lng / meter_per_degree
    degree_lat = 180.0 / pi * (2 * atan(exp(meter_lat / earth_radius)) - pi / 2.0)
    return degree_lng, degree_lat

def transfer_mercator_to_key(meter_lng, meter_lat, length):
    region_lng = int(meter_lng / length)
    region_lat = int(meter_lat / length)
    return '%s_%s' % (region_lng, region_lat)

def transfer_key_to_mercator(uid, length):
    region_lng = int(uid.split('_')[0])
    region_lat = int(uid.split('_')[1])
    meter_lng = region_lng * length
    meter_lat = region_lat * length
    return meter_lng, meter_lat

def transfer_key_to_lnglat(uid, length):
    meter_lng, meter_lat = transfer_key_to_mercator(uid, length)
    return transfer_mercator_to_lnglat(meter_lng, meter_lat)

def transfer_lnglat_to_key(degree_lng, degree_lat, length):
    meter_lng, meter_lat = transfer_lnglat_to_mercator(degree_lng, degree_lat)
    return transfer_mercator_to_key(meter_lng, meter_lat, length)

city = "beijing"
Len = 50 #精度，越小越准，但是50米已经够了

## 经纬度边界，视情况而改变
lng_ld = 116.20
lat_ld = 39.75
lng_rd = 116.58
lat_rd = 39.75
lng_lu = 116.20
lat_lu = 40.07
lng_ru = 116.58
lat_ru = 40.07


grid_ld = transfer_lnglat_to_key(lng_ld,lat_ld,Len)
grid_rd = transfer_lnglat_to_key(lng_rd,lat_rd,Len)
grid_lu = transfer_lnglat_to_key(lng_lu,lat_lu,Len)
grid_ru = transfer_lnglat_to_key(lng_ru,lat_ru,Len)

print transfer_key_to_lnglat(grid_ld,Len)
print transfer_key_to_lnglat(grid_rd,Len)
print transfer_key_to_lnglat(grid_lu,Len)
print transfer_key_to_lnglat(grid_ru,Len)
print grid_ld
print grid_rd
print grid_lu
print grid_ru

grids = json.load(open("region_" + city +"_tencent2.json"))

def isInsidePolygon(pt, poly): #point, boundary, if in return true
	c = False
	i = -1
	l = len(poly)
	j = l - 1
	while i < l-1:
		i += 1
		#print i,poly[i], j,poly[j]
		if ((poly[i][0] <= pt[0] and pt[0] < poly[j][0]) or (poly[j][0] <= pt[0] and pt[0] < poly[i][0])):
			if (pt[1] < (poly[j][1] - poly[i][1]) * (pt[0] - poly[i][0]) / (poly[j][0] - poly[i][0]) + poly[i][1]):
				c = not c
		j = i
	return c

def get_grid_id(point, grids):
	"""finding the nearest square id by hierarchical search"""
	dis_cents = 100
	gc_id = 0

	for i, gc in enumerate(grids["grid_region"]):
		dis = sqrt((float(point[0]) - float(gc[0])) ** 2 + (float(point[1]) - float(gc[1])) ** 2)
		if dis < dis_cents:
			dis_cents = dis
			gc_id = i

	gd_id = -1
	for j, gd in enumerate(grids["grid_boundary"][str(gc_id)]):
		boundary = grids["grid_boundary"][str(gc_id)][gd]
		if isInsidePolygon((float(point[0]),float(point[1])),boundary):
			gd_id = gd
			break   
	if(gd_id>0):
		return str(gc_id) + '-' + str(gd_id)
	else:
		return '-'
            
lng_begin = int(grid_ld.split('_')[0])
lng_end = int(grid_rd.split('_')[0])
lat_begin = int(grid_ld.split('_')[1])
lat_end = int(grid_lu.split('_')[1])

validation = np.zeros([lng_end-lng_begin+1,lat_end-lat_begin+1],dtype=int)

grid2region = {}


for i in range (lng_begin,lng_end+1):
    print(i)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))#现在
    for j in range (lat_begin,lat_end+1):
        grid_id = str(i)+'_'+str(j)
        grid_center = transfer_key_to_lnglat(grid_id,Len)
        #print grid_id,grid_center
        region_id=get_grid_id(grid_center,grids)
        if (region_id == '-'):
            validation[i-lng_begin][j-lat_begin]=676
        else:
            validation[i-lng_begin][j-lat_begin]=int(region_id.split('-')[1])
        grid2region[grid_id]=region_id


json.dump(grid2region, open("grid_region_" + city +"_tencent.json","w"))  






