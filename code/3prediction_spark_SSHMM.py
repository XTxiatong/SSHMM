##use 1 day data to test 
## load the model
##indetify cadidata pool 5 timeslice data
## may need use clusterring infomation, otherwise the accracy is low


from numpy import *
import math
import random
import scipy.io as sio
import time
import os
import sys
from pyspark import SparkContext,SparkConf

def extendx(index,x_3): #change from 0 to X
	global O,A,Pi,mu,sigma,assign
	x_3=O[index,:,:].reshape([O.shape[1],O.shape[2]])
	return(index,x_3)

def normfun(x,Mu,Sigma):
	ss=sqrt(Sigma)
	temp=(x-Mu)/ss
	p=exp(-temp**2/2)/sqrt(2*math.pi)/ss
	return p

def calcForward(r,X):
	global O,A,Pi,mu,sigma,assign
	alpha=zeros([len(X),K]) #W*K
	print 'prediction sequence length:', alpha.shape
	#n=0
	for k in range(K):
		o=1
		for l in range (L):
			o=o*normfun(X[0][l],mu[k][l],sigma[k][l])
		alpha[0][k]=Pi[r][k]*o
	#n=1,2,...
	for n in range (1,len(X)):
		for k in range (K):
			o=1
			for l in range (L):
				o=o*normfun(X[n][l],mu[k][l],sigma[k][l])
			alpha[n][k]=o*dot(alpha[n-1,:].reshape(1,K),A[r,:,k].reshape(K,1))
	#print 'po_alpha'
	#print sum(alpha[len(X)-1,:])
	
	beta=zeros([len(X),K])
	#n=N-1
	beta[-1,:]=1
					
	#n=N-2,N-3,...0
	for n in range (W-2,-1,-1):
		for k in range (K):
			s=0
			for j in range (K):
				o=1
				for l in range (L):
					o=o*normfun(X[n+1][l],mu[j][l],sigma[j][l])
				s=s+A[r][k][j]*beta[n+1][j]*o
			beta[n][k]=s
	#P(O|^)
	#print 'po_beta'
	po=0
	for k in range (K):
		o=1
		for l in range (L):
			o=o*normfun(X[0][l],mu[k][l],sigma[k][l])
		po=po+Pi[r][k]*beta[0][k]*o
	#print po		
	
	gamma=zeros([N,K])
	#gamma[n][k]=alpha[n][k]*beta[n][k]/po
	gamma=alpha*beta/po
		
	return gamma

def calcGamma(r,X):
	global O,A,Pi,mu,sigma,assign
	alpha=zeros([len(X),K]) #W*K
	print 'prediction sequence length:', alpha.shape
	#n=0
	for k in range(K):
		o=1
		for l in range (L):
			o=o*normfun(X[0][l],mu[k][l],sigma[k][l])
		alpha[0][k]=Pi[r][k]*o
	#n=1,2,...
	for n in range (1,len(X)):
		for k in range (K):
			o=1
			for l in range (L):
				o=o*normfun(X[n][l],mu[k][l],sigma[k][l])
			alpha[n][k]=o*dot(alpha[n-1,:].reshape(1,K),A[r,:,k].reshape(K,1))
	#print 'po_alpha'
	#print sum(alpha[len(X)-1,:])
	
	beta=zeros([len(X),K])
	#n=N-1
	beta[-1,:]=1
					
	#po
	po = sum(alpha[len(X)-1,:])

	gamma=zeros([N,K])
	#gamma[n][k]=alpha[n][k]*beta[n][k]/po
	gamma=alpha*beta/po
		
	return gamma

def calcOP(state,obser):
	global O,A,Pi,mu,sigma,assign
	#print obser.shape
	o=1
	for l in range (L):
		o=o*normfun(obser[l],mu[state][l],sigma[state][l])
	return o

def pre_state(r,X):
	global O,A,Pi,mu,sigma,assign
	prediction = zeros([1,N-T,L])
	for n in range (T,N):
		print 'n:',n
		gamma=calcGamma(r,O[r,n-W:n,:])
		temg=gamma[-1,:]
		thisstate=argmax(temg)
		nextstate=argmax(A[r,thisstate,:])
		prediction[0,n-T,:] = mu[nextstate,:]
	return (r,prediction)

def extraP(x):
	prediction_all = zeros([R,N-T,L])
	for item in x[1]:
		r = item[0]
		prediction_all[r,:,:]=item[1]
	return prediction_all

if __name__ == "__main__":
	
	root = '/home/data/data/usera/xiatong/HMM/o_1hours_31day.npy'
	O = load(root)
	O = O[:,:,:]
	A=load('../result/result_R665_504_N24_S100max_iter5_pi1_sigmafix/A0_iter0.npy')
	Pi=loadtxt('../result/result_R665_504_N24_S100max_iter5_pi1_sigmafix/pi0_iter0')
	mu=loadtxt('../result/result_R665_504_N24_S100max_iter5_pi1_sigmafix/mu0_iter0')
	sigma=loadtxt('../result/result_R665_504_N24_S100max_iter5_pi1_sigmafix/sigma0_iter0')

	R=O.shape[0]
	N=O.shape[1]-24 # the total length of data
	L=O.shape[2]
	K=100
	T=24*21 # the observation for train
	W = 24
	print 'R,N,L,T',R,N,L,T

	
	conf = SparkConf().setAppName("xiatong:HMM").setMaster('yarn')
	conf.set('spark.yarn.queue','usera')
	conf.set('spark.executor.memory','6g')
	conf.set('spark.driver.memory','6g')
	conf.set('spark.executor.instances','100')
	sc = SparkContext(conf=conf)
	sc.setLogLevel("ERROR")
	print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'Estep:====================================')
	#print XRDD.getNumPartitions()
	XRDD = sc.parallelize(zeros([R,L])).zipWithIndex().map(lambda x:extendx(x[1],x[0])) #for training 
	SRDD = XRDD.map(lambda x:pre_state(x[0],x[1]))
	print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'Mstep:====================================')
	PRDD = SRDD.map(lambda x: (1,(x[0],x[1]))).groupByKey().map(lambda x: extraP(x))
	prediction = zeros([R,N-T,L]) #the last column should be 0
	prediction = PRDD.collect()
	sio.savemat('prediction_result_R665_504_N24_S100max_iter5_pi1_sigmafix_iter0.mat',{'prediction':prediction[0]}) 

