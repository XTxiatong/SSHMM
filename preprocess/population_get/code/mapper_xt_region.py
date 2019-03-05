# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import time
import json
import math
from collections import Counter
import datetime
from math import log, tan, sin, cos, atan, sqrt, exp



earth_radius = 6378137.0
pi = 3.1415926535897932384626
meter_per_degree = earth_radius * pi / 180.0
Len = 50
lng_ld = 116.20
lat_ld = 39.75
lng_rd = 116.58
lat_rd = 39.75
lng_lu = 116.20
lat_lu = 40.07
lng_ru = 116.58
lat_ru = 40.07

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

def read_from_text(file):
	for line in file:
		yield line.strip('\r\n').split('\t')

def load_grids():
	# file path needed to be modified before running
	return json.load(open("code/grid_region_tencent_50m.json"))

def get_grid_id(point):
	if (lng_ld <= point[0] <= lng_rd and lat_ld <= point[1] <= lat_lu):
		grid_id = transfer_lnglat_to_key(point[0], point[1], Len)
		return grid_id
	else:
		return '-'
		
def move_count(gid, gid_b, tp, tp_b, flow):
	if gid == gid_b:  # no moving, no flow count
		pass
	else:
		flow.append((tp_b, gid_b, -1))  # -1: pass out
		flow.append((tp, gid, 1))  # 1: pass in
	return flow
	
def points_filter(trace, grids, THE=30, INTER=False, FLOW=False):
	"""construct, filter and interpolate trajectory"""

	assert 1 <= THE <= 60, "threshold(THE) should fall in [1,60] "
	# conform the start point of time
	tim_start = int(time.mktime(time.strptime("2018-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")))

	# basic settings
	DAY_START = 60 / THE * 6  # e.g., start from 6 a.m., when THE=30, DAY_START=12
	TIM_SLOTS = 60 / THE * 24  # e.g., 24 hours in a day, when THE=30, TIM_SLOTS=48
	MAX_WAITS = 60 / THE * 2  # e.g., 3 hours as maximum interval, when THE=30, MAX_WAITS=6
	# MIN_POINTS_DAY = 60 / THE * 9  # e.g., trajectory records should last at least half day (9/(24-6)=1/2).

	# read data and construct trajectory
	uid = trace[0]
	points = trace[1].split(";")

	points_tmp = {}
	for p in points:
		#lon, lat, tim = p.split(",")
		tim, lon, lat = p.split(",")
		#tim_stamp = time.mktime(time.strptime(tim, "%Y-%m-%d %H:%M:%S"))
		tim_stamp = tim
		tim = (int(tim_stamp) - tim_start) / 60 / THE
		lon = float(lon)
		lat = float(lat)

		gid = get_grid_id((lon, lat))
		#print(gid)
		if(gid!='-'):
			if tim not in points_tmp:
				points_tmp[tim] = [gid]
			else:
				points_tmp[tim].append(gid)

	# find the most common location during the time slot
	points_grids = []
	for tp in points_tmp:
		gid = Counter(points_tmp[tp]).most_common()[0][0]
		points_grids.append((tp, gid, 1))

	# sort trajectory by time order
	points_grids = sorted(points_grids, key=lambda x: x[0], reverse=False)

	points_region = []
	if not INTER:
		for item in points_grids:
			tp, gid, count = item[0],item[1],item[2]
			if (gid in grids):
				if (grids[gid]!='-'):
					points_region.append((tp,grids[gid],1))
		if not FLOW:
			return points_region
			
		else:
			points_region_flow = []
			for i in range(1, len(points_region)):
				(tp, gid_n, _), (tp_b, gid_b, _) = points_region[i], points_region[i - 1]
				points_region_flow = move_count(gid=gid_n, gid_b=gid_b, tp=tp, tp_b=tp_b,
													   flow=points_region_flow)
			return points_region_flow
				
		
	# preparation for interpolation: find most common locations array
	common_tmp = {}
	common_locations = ['10000-10000'] * TIM_SLOTS
	for i in range(len(points_grids)):
		tp, gid, _ = points_grids[i]
		tid = tp % TIM_SLOTS
		if tid not in common_tmp:
			common_tmp[tid] = [gid]
		else:
			common_tmp[tid].append(gid)
	for tid in range(TIM_SLOTS):
		if tid in common_tmp:
			home = Counter(common_tmp[tid]).most_common()[0][0]
			common_locations[tid] = home
	#print(common_locations)

	# start personal trajectory interpolation
	points_grids_interp = []
	points_grids_flow = []
	for i in range(1, len(points_grids)):
		(tp, gid_n, _), (tp_b, gid_b, _) = points_grids[i], points_grids[i - 1]
		gid_coor =  transfer_key_to_lnglat(gid_n, Len)
		gid_b_coor = transfer_key_to_lnglat(gid_b, Len)
		tlen = tp - tp_b
			
		# during the night with lot of missing records: the interpolation is based on the location visit history
		if ((tp % TIM_SLOTS) < DAY_START or (tp_b % TIM_SLOTS) < DAY_START) and int(TIM_SLOTS / 2) > tlen > MAX_WAITS:
			for ti in range(tp_b, tp): #add, interpolate from the first point
				gid_common = common_locations[ti % TIM_SLOTS]
				points_grids_interp.append((ti, gid_common, 1))
		# during the day: the trajectory interpolation is based on the neighbors with linear interpolation
		elif tlen <= MAX_WAITS:
			tp_b2 = tp_b
			for ti in range(0, abs(tlen)):
				x_i = gid_b_coor[0] + ti / float(tlen) * (gid_coor[0] - gid_b_coor[0])
				y_i = gid_b_coor[1] + ti / float(tlen) * (gid_coor[1] - gid_b_coor[1])
				#assert x_i > 100, "coordinates order: longitude, latitude"
				# if interpolation point is not in the boundary(116.10-116.70;39.70-40.20), ignore it
				gid_interp = get_grid_id((x_i, y_i))
				if(gid_interp !='-'):
					points_grids_interp.append((tp_b + ti, gid_interp, 1))

		# if too many missing records, we give up to interpolate trajectory
		else:
			points_grids_interp.append((tp_b, gid_b, 1))
		if(i==(len(points_grids)-1)): #add, for the last point, we still need it!
			points_grids_interp.append((tp, gid_n, 1))

	if INTER:
		points_region_inter = []
		for item in points_grids_interp:
			tp, gid, count = item[0],item[1],item[2]
			if (gid in grids):
				if (grids[gid]!='-'):
					points_region_inter.append((tp,grids[gid],1))
		if not FLOW:
			return points_region_inter
		else:
			points_region_inter_flow = []
			for i in range(1, len(points_region_inter)):
				(tp, gid_n, _), (tp_b, gid_b, _) = points_region_inter[i], points_region_inter[i - 1]
				points_region_inter_flow = move_count(gid=gid_n, gid_b=gid_b, tp=tp, tp_b=tp_b,
													   flow=points_region_inter_flow)
			return points_region_inter_flow

def main(THE=30, INTER=False, FLOW=False):
	grids = load_grids()
	#print(grids)
	for line in read_from_text(sys.stdin):
		points_grids = points_filter(line, grids, THE=THE, INTER=INTER, FLOW=FLOW)
		for p in points_grids:
			print("%s\t%d" % (str(p[0] * p[2]) + "=" + str(p[1]), p[2]))


if __name__ == "__main__":
	temporal_threshold = int(sys.argv[1])
	is_interpolation = int(sys.argv[2]) == 1
	is_flow = int(sys.argv[3]) == 1
	main(THE=temporal_threshold, INTER=is_interpolation, FLOW=is_flow)





