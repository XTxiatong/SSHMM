# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:36:30 2018

@author: 111

pattern 的可视化

选几个街道
state的颜色是原始颜色，
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from pylab import *  
  
params = 'mu'
root = 'result//result_R665_504_N24_S100max_iter5_pi1_sigmafix//'
num = 0 # num of id
ite = 1
mu = np.loadtxt(root+'mu'+str(num)+'_iter'+str(ite))
state = np.loadtxt(root+'predict_stata'+str(num)+'_iter'+str(ite))


# #############################################################################
# Plot result
from matplotlib import cm 
from matplotlib import axes
import seaborn as sns
from pandas import DataFrame
sns.set(rc={"font.size":16,"axes.labelsize":16})
label = np.loadtxt('clustering\\labels',dtype = int)
labels_unique = np.unique(label)
N = len(labels_unique)
mediods = np.loadtxt('clustering\\clusters8\\mediods0.txt',dtype = int)
clusters = np.loadtxt('clustering\\clusters8\\clusterAssignments0.txt',dtype = int)
XINDEX=[579,401,46,378,410,363,540,507,560,137,103,499,326,522,174,505,210,173,217,274];
XINDEX=[576,398,45,376,407,361,537,504,557,135,101,496,324,519,172,502,208,171,215,272];

assign=[4,3,3,3,3,3,8,8,4,1,7,4,8,2,2,2,6,5,5,5];
#for i in range (2):
i = 12
c = assign[i]
regions_c = np.where(assign==c)[0]
regions = XINDEX[i]
regions = 129
region_state = state[regions,:].reshape([3*7,24])
#变成类
region_state_s = np.zeros([region_state.shape[0],region_state.shape[1]])
for i in range(region_state.shape[0]):
    for j in range (region_state.shape[1]):
        region_state_s[i][j] = label[int(region_state[i][j])]
        
weekend = [1,5,6,7,14,15,21]
weekday = [2,3,4,8,9,10,11,12,13,16,17,18,19,20]
region_state_s_n = np.zeros([len(weekend),region_state.shape[1]],dtype=int)
region_state_s_w = np.zeros([len(weekday),region_state.shape[1]],dtype=int)
for i in range (len(weekend)):
    region_state_s_n[i,:] = region_state[weekend[i]-1,:]
#region_state_s_n[i,22]=0     
#region_state_s_n[i,23]=99 #为了统一可视化尺度
for i in range (len(weekday)):
    region_state_s_w[i,:] = region_state[weekday[i]-1,:]
#region_state_s_w[i,22]=0
#region_state_s_w[i,23]=99 #为了统一可视化尺度
#可视化
f, ax= plt.subplots(figsize = (12, 0.4*7),dpi=100)
d_inde=['1(Sun.)','5(Thu.)','6(Fri.)','7(Sat.)','14(Sat.)','15(Sun.)','21(Sat.)']
tim=[str(i) for i in range(24)]
df=DataFrame(region_state_s_n,index=d_inde,columns=tim)
sns.heatmap(df,annot=True, fmt="d",  cmap="RdBu_r",linewidths = 0.3, ax = ax)
ax.tick_params(axis='x',labelsize=20, colors='black', labeltop=False, labelbottom=True) # x轴
ax.tick_params(axis='y',labelsize=20, colors='black') # y轴
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
ax.set_title('Weekends Patern' + str(c) +'_'+ str(regions),size=20)
ax.set_xlabel('Time/hour',size=20)
ax.set_ylabel('Date',size=20)
f.show() 
f, ax= plt.subplots(figsize = (12, 0.4*14),dpi=100)
d_inde=['2(Mon.)','3(Tue.)','4(Wed.)','8(Fri.)','9(Mon.)','10(Tue.)','11(Wed.)','12(Thu.)','13(Fri.)','16(Mon.)','17(Tue.)','18(Wed.)','19(Thu.)','20(Fri.)',]
tim=[str(i) for i in range(24)]
df=DataFrame(region_state_s_w,index=d_inde,columns=tim)
sns.heatmap(df, annot=True, fmt="d", cmap="RdBu_r",linewidths = 0.3, ax = ax)
ax.tick_params(axis='x',labelsize=20, colors='black', labeltop=False, labelbottom=True) # x轴
ax.tick_params(axis='y',labelsize=20, colors='black') # y轴
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
ax.set_title('Weekdays Patern' + str(c),size=20)
ax.set_xlabel('Time/hour',size=20)
ax.set_ylabel('Date',size=20)
f.show() 


"""
Created on Wed Aug 22 16:00:40 2018

@author: 111

选状态可视化
"""

import matplotlib.pyplot as plt
import numpy as np 
from sklearn.cluster import MeanShift, estimate_bandwidth

params = 'mu'
root = 'result//result_R665_504_N24_S100max_iter5_pi1_sigmafix//'
num = 0 # num of id
ite = 1
mu = np.loadtxt(root+'mu'+str(num)+'_iter'+str(ite))
state = np.loadtxt(root+'predict_stata'+str(num)+'_iter'+str(ite),dtype = int)
state=state[:,0:504]

#找出现次数最多的
from collections import Counter
state_fre = Counter(state[regions,:].tolist()).most_common(10) # 返回出现频率最高的两个数
index = [x[0] for x in state_fre]
# #############################################################################
# Plot result
import seaborn as sns
from pandas import DataFrame
sns.set(rc={"font.size":16,"axes.labelsize":16})

#把状态取出来
a = mu[index,:]
d_inde=[]
w = 0.3*len(index)
for i in index:
    d_inde.append('State '+str(i))
poi= ['Restaurant', 'Company', 'Agency', 'Shopping', 'Service', 'Entertainment','Attractions', 'Education', 'Residence','Arrving','Leaving','Staying']
f, ax= plt.subplots(figsize = (12, w))
df=DataFrame(a,index=d_inde,columns=poi)
sns.heatmap(df,cmap='Blues', linewidths = 0.3, ax = ax)
ax.tick_params(axis='x',labelsize=20, colors='black', labeltop=True, labelbottom=False) # x轴
ax.tick_params(axis='y',labelsize=18, colors='black') # y轴
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
#ax.set_xlabel(state_count[c][0]+','+state_count[c][1]+','+state_count[c][2]+','+state_count[c][3]+','+state_count[c][4],size=25)
#f.savefig('clustering\\sns_style_update.jpg', dpi=300, bbox_inches='tight')
f.show() 
# #############################################################################
# Plot result
from collections import Counter
state_fre = Counter(state[regions,:].tolist()).most_common(5) # 返回出现频率最高的两个数
index = [x[0] for x in state_fre]
import seaborn as sns
from pandas import DataFrame
sns.set(rc={"font.size":16,"axes.labelsize":16})

#把状态取出来
a = mu[index,:]
d_inde=[]
w = 0.3*len(index)
for i in index:
    d_inde.append('State '+str(i))
poi= ['Restaurant', 'Company', 'Agency', 'Shopping', 'Service', 'Entertainment','Attractions', 'Education', 'Residence','Arrving','Leaving','Staying']
f, ax= plt.subplots(figsize = (12, w),dpi=100)
df=DataFrame(a,index=d_inde,columns=poi)
sns.heatmap(df,cmap='Blues', linewidths = 0.3, ax = ax)
ax.tick_params(axis='x',labelsize=20, colors='black', labeltop=True, labelbottom=False) # x轴
ax.tick_params(axis='y',labelsize=18, colors='black') # y轴
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
#ax.set_xlabel(state_count[c][0]+','+state_count[c][1]+','+state_count[c][2]+','+state_count[c][3]+','+state_count[c][4],size=25)
#f.savefig('clustering\\sns_style_update.jpg', dpi=300, bbox_inches='tight')
f.show() 


#把状态取出来
a = np.hstack((mu[index,9:12],mu[index,0:9]))
d_inde=[]
w = 0.3*len(index)
for i in index:
    d_inde.append('State '+str(i))
poi= ['Arrving','Leaving','Staying','Restaurant', 'Company', 'Agency', 'Shopping', 'Service', 'Entertainment','Attractions', 'Education', 'Residence']
f, ax= plt.subplots(figsize = (12, w),dpi=100)
df=DataFrame(a,index=d_inde,columns=poi)
sns.heatmap(df,cmap='Blues', linewidths = 0.3, ax = ax)
ax.tick_params(axis='x',labelsize=20, colors='black', labeltop=True, labelbottom=False) # x轴
ax.tick_params(axis='y',labelsize=18, colors='black') # y轴
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
#ax.set_xlabel(state_count[c][0]+','+state_count[c][1]+','+state_count[c][2]+','+state_count[c][3]+','+state_count[c][4],size=25)
#f.savefig('clustering\\sns_style_update.jpg', dpi=300, bbox_inches='tight')
f.show() 

weekend = [1,5,6,7]
weekday = [2,3,4,8,9]
region_state_s_n = np.zeros([len(weekend),region_state.shape[1]],dtype=int)
region_state_s_w = np.zeros([len(weekday),region_state.shape[1]],dtype=int)
for i in range (len(weekend)):
    region_state_s_n[i,:] = region_state[weekend[i]-1,:]
#region_state_s_n[i,22]=0     
#region_state_s_n[i,23]=99 #为了统一可视化尺度
for i in range (len(weekday)):
    region_state_s_w[i,:] = region_state[weekday[i]-1,:]
#region_state_s_w[i,22]=0
#region_state_s_w[i,23]=99 #为了统一可视化尺度
#可视化
f, ax= plt.subplots(figsize = (12, 0.4*len(weekend)),dpi=100)
d_inde=['1(Sun.)','5(Thu.)','6(Fri.)','7(Sat.)']
tim=[str(i) for i in range(24)]
df=DataFrame(region_state_s_n,index=d_inde,columns=tim)
sns.heatmap(df,annot=True, fmt="d",  cmap="RdBu_r",linewidths = 0.3, ax = ax)
ax.tick_params(axis='x',labelsize=20, colors='black', labeltop=False, labelbottom=True) # x轴
ax.tick_params(axis='y',labelsize=20, colors='black') # y轴
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
#ax.set_title('Weekends Patern' + str(c) +'_'+ str(regions),size=20)
ax.set_title('Non-working days',size=20)
ax.set_xlabel('Time/hour',size=20)
ax.set_ylabel('Date',size=20)
f.show() 
f, ax= plt.subplots(figsize = (12, 0.4*len(weekday)),dpi=100)
d_inde=['2(Mon.)','3(Tue.)','4(Wed.)','8(Fri.)','9(Mon.)']
tim=[str(i) for i in range(24)]
df=DataFrame(region_state_s_w,index=d_inde,columns=tim)
sns.heatmap(df, annot=True, fmt="d", cmap="RdBu_r",linewidths = 0.3, ax = ax)
ax.tick_params(axis='x',labelsize=20, colors='black', labeltop=False, labelbottom=True) # x轴
ax.tick_params(axis='y',labelsize=20, colors='black') # y轴
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
#ax.set_title('Weekdays Patern' + str(c),size=20)
ax.set_title('Working days',size=20)
ax.set_xlabel('Time/hour',size=20)
ax.set_ylabel('Date',size=20)
f.show() 


weekend = [1,5,6,7]
weekday = [16,17,18,19,20]
region_state_s_n = np.zeros([len(weekend),region_state.shape[1]],dtype=int)
region_state_s_w = np.zeros([len(weekday),region_state.shape[1]],dtype=int)
for i in range (len(weekend)):
    region_state_s_n[i,:] = region_state[weekend[i]-1,:]
#region_state_s_n[i,22]=0     
#region_state_s_n[i,23]=99 #为了统一可视化尺度
for i in range (len(weekday)):
    region_state_s_w[i,:] = region_state[weekday[i]-1,:]
#region_state_s_w[i,22]=0
#region_state_s_w[i,23]=99 #为了统一可视化尺度
#可视化
f, ax= plt.subplots(figsize = (12, 0.4*len(weekend)),dpi=100)
d_inde=['1(Sun.)','5(Thu.)','6(Fri.)','7(Sat.)']
tim=[str(i) for i in range(24)]
df=DataFrame(region_state_s_n,index=d_inde,columns=tim)
sns.heatmap(df,annot=True, fmt="d",  cmap="RdBu_r",linewidths = 0.3, ax = ax)
ax.tick_params(axis='x',labelsize=20, colors='black', labeltop=False, labelbottom=True) # x轴
ax.tick_params(axis='y',labelsize=20, colors='black') # y轴
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
#ax.set_title('Weekends Patern' + str(c) +'_'+ str(regions),size=20)
ax.set_title('Non-working days',size=20)
ax.set_xlabel('Time/hour',size=20)
ax.set_ylabel('Date',size=20)
f.show() 
f, ax= plt.subplots(figsize = (12, 0.4*len(weekday)),dpi=100)
d_inde=['16(Mon.)','17(Tue.)','18(Wed.)','19(Thu.)','20(Fri.)']
tim=[str(i) for i in range(24)]
df=DataFrame(region_state_s_w,index=d_inde,columns=tim)
sns.heatmap(df, annot=True, fmt="d", cmap="RdBu_r",linewidths = 0.3, ax = ax)
ax.tick_params(axis='x',labelsize=20, colors='black', labeltop=False, labelbottom=True) # x轴
ax.tick_params(axis='y',labelsize=20, colors='black') # y轴
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=360)
#ax.set_title('Weekdays Patern' + str(c),size=20)
ax.set_title('Working days',size=20)
ax.set_xlabel('Time/hour',size=20)
ax.set_ylabel('Date',size=20)
f.show() 





