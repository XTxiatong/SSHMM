# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 00:51:08 2018
@author: xiatong
This is the pyspark code for baseline HMM
"""



from numpy import *
import numpy as np
from math import * 
import math
import scipy.io as sio
import time
import os
import sys
from pyspark import SparkContext,SparkConf

def extendx(index,x_3): #change from 0 to X
	global X,N,M,K,R,L
	x_3=X[index,:,:].reshape([X.shape[1],X.shape[2]])
	return(index,x_3)
	
def extendd(index,d_3): #change from 0 to X
	global D,N,M,K,R,L
	d_3=D[index,:,:].reshape([D.shape[1],D.shape[2]])
	return(index,d_3)

def extendabg(index,x_2): #change from N to N*K
	global N,M,K,R,L
	x_3=zeros([N,K])
	return(index,x_3)
	
def extendxi(index,xi_3): #change from (N-1)*K to (N-1)*K*K
	global N,M,K,R,L
	xi_4=zeros([N-1,K,K])
	return(index,xi_4)
	
def extendA(index,a_2): #change from (N-1)*K to (N-1)*K*K
	global N,M,K,R,L
	a_3=ones([K,K])*1.0/K # [R,K,K]
	return(index,a_3)
	
def extendmusigma(x):
	global N,M,K,R,L
	global mu_,sigma_
	input = x[0] #r
	X_r = x[1][0] #wN*L
	alpha_r = x[1][1] #N*K
	beta_r = x[1][2]  #N*K
	gamma_r = x[1][3] #N*K
	xi_r = x[1][4] #N-1*N*K
	A_r = x[1][5] #K*K
	pi_r = x[1][6] #K
	return(input,(X_r,alpha_r,beta_r,gamma_r,xi_r,A_r,pi_r,mu_,sigma_))
	
def normfun2(x,Mu,Sigma):
	ss=sqrt(Sigma)
	if ss<0.001:
		ss = 0.001
	temp=(x-Mu)/ss
	if (temp>2 or temp<-2):
		#p=0.004432
		p=0.05/ss  #approximate solution
	else:
		p=exp(-temp**2/2)/sqrt(2*math.pi)/ss
	return p
	
def normfun(x,Mu,Sigma):
	ss=sqrt(Sigma)
	temp=(x-Mu)/ss
	p=exp(-temp**2/2)/sqrt(2*math.pi)/ss
	return p

def calcAlpha(x,begin):
	global N,M,K,R,L
	input = x[0] #r
	X_r = x[1][0] #wN*L
	alpha_r = x[1][1] #N*K
	beta_r = x[1][2]  #N*K
	gamma_r = x[1][3] #N*K
	xi_r = x[1][4] #N-1*N*K
	A_r = x[1][5] #K*K
	pi_r = x[1][6] #K
	mu = x[1][7]
	sigma = x[1][8]

	#n=0, the first slot in this segment
	for r in [input]:
		for k in range(K):
			o=1
			for l in range(0,L):
				o = o * normfun(X_r[begin][l],mu[k][l],sigma[k][l])
			alpha_r[0][k]=pi_r[k]*o

	#n=1,2,...
	for r in [input]:
		for n in range (1,N):
			for k in range (K):
				o=1
				for l in range(0,L):
					o = o * normfun(X_r[n+begin][l],mu[k][l],sigma[k][l])
				alpha_r[n][k]=o*np.dot(alpha_r[n-1,:].reshape(1,K),A_r[:,k].reshape(K,1)) #street r prob of transition i->j: A[r,i,j]
				
	#P(O|^)	
	po = sum(alpha_r[N-1,:])
	return (input,(X_r,alpha_r,beta_r,gamma_r,xi_r,A_r,pi_r,mu,sigma))
	
def calcBeta(x,begin):
	global N,M,K,R,L
	global mu,sigma
	input = x[0] #r
	X_r = x[1][0] #wN*L
	alpha_r = x[1][1] #N*K
	beta_r = x[1][2]  #N*K
	gamma_r = x[1][3] #N*K
	xi_r = x[1][4] #N-1*N*K
	A_r = x[1][5] #K*K
	pi_r = x[1][6] #K
	mu = x[1][7]
	sigma = x[1][8]
	
	#n=N-1
	beta_r[-1,:]=1
	
	#n=N-2,N-3,...0
	for r in [input]:
		for n in range (N-2,-1,-1):
			for k in range (K):
				s=0
				for j in range (K):
					if A_r[k][j]>1e-30:
						o=1
						for l in range(0,L):
							o = o * normfun(X_r[n+1+begin][l],mu[j][l],sigma[j][l])
						s=s+A_r[k][j]*beta_r[n+1][j]*o
				beta_r[n][k]=s
	#P(O|^)
	po=0
	for k in range (K):
		o=1
		for l in range(0,L):
			o = o * normfun(X_r[begin][l],mu[k][l],sigma[k][l])
		po = po + pi_r[k]*beta_r[0][k]*o
	return (input,(X_r,alpha_r,beta_r,gamma_r,xi_r,A_r,pi_r,mu,sigma))
	
def calcGamma(x,begin):
	global N,M,K,R,L
	input = x[0] #r
	X_r = x[1][0] #wN*L
	alpha_r = x[1][1] #N*K
	beta_r = x[1][2]  #N*K
	gamma_r = x[1][3] #N*K
	xi_r = x[1][4] #N-1*N*K
	A_r = x[1][5] #K*K
	pi_r = x[1][6] #K
	mu = x[1][7]
	sigma = x[1][8]
	for r in [input]:
		po=sum(alpha_r[N-1,:])
		if po == 0:
			print('po=0!')
			po = 1e-6
		for n in range (N):
			for k in range (K):   #the probability that observartion at n-th position is state k 
				gamma_r[n][k]=alpha_r[n][k]*beta_r[n][k]/po
	return(input,(X_r,alpha_r,beta_r,gamma_r,xi_r,A_r,pi_r,mu,sigma))
	
def calcXi(x,begin):
	global N,M,K,R,L
	input = x[0] #r
	X_r = x[1][0] #wN*L
	alpha_r = x[1][1] #N*K
	beta_r = x[1][2]  #N*K
	gamma_r = x[1][3] #N*K
	xi_r = x[1][4] #N-1*N*K
	A_r = x[1][5] #K*K
	pi_r = x[1][6] #K
	mu = x[1][7]
	sigma = x[1][8]
	for r in [input]:
		po=sum(alpha_r[N-1,:])
		if po == 0:
			print('po=0!')
			po = 1e-6
		for n in range (N-1):
			for j in range (K):
				for k in range (K):#the probability that observartion at (n+1)-th position is state k ,at n-th position is state j
					if A_r[k][j]>1e-30:
						o = 1
						for l in range(0,L):
							o = o * normfun(X_r[n+1+begin][l],mu[k][l],sigma[k][l])
						xi_r[n][j][k]=alpha_r[n][j]*A_r[j][k]*beta_r[n+1][k]*o/po
	return(input,(X_r,alpha_r,beta_r,gamma_r,xi_r,A_r,pi_r,mu,sigma))
	
def updatePi(x,begin):
	global N,M,K,R,L
	input = x[0] #r
	X_r = x[1][0] #wN*L
	alpha_r = x[1][1] #N*K
	beta_r = x[1][2]  #N*K
	gamma_r = x[1][3] #N*K
	xi_r = x[1][4] #N-1*N*K
	A_r = x[1][5] #K*K
	pi_r = x[1][6] #K
	mu = x[1][7]
	sigma = x[1][8]
	if(begin == 0):
		for k in range (K):
			pi_r[k]= gamma_r[0,k]
	return(input,(X_r,alpha_r,beta_r,gamma_r,xi_r,A_r,pi_r,mu,sigma))
	
def updateA(x,begin):
	global N,M,K,R,L
	input = x[0] #r
	X_r = x[1][0] #wN*L
	alpha_r = x[1][1] #N*K
	beta_r = x[1][2]  #N*K
	gamma_r = x[1][3] #N*K
	xi_r = x[1][4] #N-1*N*K
	A_r = x[1][5] #K*K
	pi_r = x[1][6] #K
	mu = x[1][7]
	sigma = x[1][8]
	for j in range (K):
		Sitaj=sum(xi_r[:,j,:])
		if Sitaj == 0:
			pass
		else:
			for k in range (K):   #Xi:shape on dim2: N-1
				A_r[j][k]=sum(xi_r[:,j,k])/Sitaj
	return(input,(X_r,alpha_r,beta_r,gamma_r,xi_r,A_r,pi_r,mu,sigma))
	

def updateMuSigma(x,begin):
	global X,D,N,M,K,R,L
	input = x[0] #r
	X_r = x[1][0] #wN*L
	alpha_r = x[1][1] #N*K
	beta_r = x[1][2]  #N*K
	gamma_r = x[1][3] #N*K
	xi_r = x[1][4] #N-1*N*K
	A_r = x[1][5] #K*K
	pi_r = x[1][6] #K
	mu = x[1][7]
	sigma = x[1][8]
	for k in range (K):
		Gamma_k = sum(gamma_r[:,k])
		if Gamma_k == 0:
			print('Gamma_k=0!')
			pass
		else:
			for l in range(0,L):
				Gamma_rn = gamma_r[:,k]
				X_rn = X_r[begin:begin+N,l]
				mu[k,l] = sum(Gamma_rn*X_rn)/Gamma_k
				#X_rn2=(X_rn-mu[k][l])**2
				#sigma[k,l]=sum(Gamma_rn*X_rn2)/Gamma_k
	return (input,(X_r,alpha_r,beta_r,gamma_r,xi_r,A_r,pi_r,mu,sigma))
	
def viterbi(x):
	global N,M,K,R,L

	input = x[0] #r
	#X_r = x[1][0][0] #wN*L
	alpha_r = x[1][0][1] #N*K
	beta_r = x[1][0][2]  #N*K
	gamma_r = x[1][0][3] #N*K
	xi_r = x[1][0][4] #N-1*N*K
	A_r = x[1][0][5] #K*K
	pi_r = x[1][0][6] #K
	mu = x[1][0][7] #K
	sigma = x[1][0][8] #K
	D_r = x[1][1] #totalN*K
	
	delta=zeros([M,K])
	psi=zeros([M,K])
	psi_k=zeros([K,1])
	Z_temp=zeros([1,M],dtype=int) #state for one day
	N2 = D_r.shape[0]
	Z=zeros([1,N2],dtype=int) #all state
	for begin in range(0,N2,M):
		#print("decode: begin", begin, 'end', begin + M - 1)
		for k in range (K):   #init 
			o = 1
			for l in range(L):
				o = o * normfun(D_r[begin][l],mu[k][l],sigma[k][l])
			delta[0][k]=pi_r[k]*o
			psi[0][k]=0
		for n in range (1,M):   #n=1,2,...
			for k in range (K):
				for j in range (K):
					psi_k[j]=delta[n-1][j]*A_r[j][k]
				j=argmax(psi_k)
				psi[n][k]=j
				o = 1
				for l in range(L):
					o = o * normfun(D_r[n+begin][l],mu[k][l],sigma[k][l])
				#print o
				delta[n][k]=delta[n-1][j]*A_r[j][k]*o
		Z_temp[0,M-1]=argmax(delta[M-1,:])  #end
		for n in range (M-2,-1,-1):    #backfard
			Z_temp[0,n]=psi[n+1][Z_temp[0,n+1]]
		Z[0][begin:begin+M]=Z_temp
	return (input,Z)
	
def extraSate(x):
	global D,N,M,K,R,L
	N2 = D.shape[1]
	state_all = zeros([R,N2]) #total length
	for item in x[1]:
		r = item[0]
		state_r = item[1]
		state_all[r,:] = state_r
	return state_all
	
if __name__ == "__main__":

	if len(sys.argv) != 2:
		print '<in 1>'
		exit(1)
	in1 = sys.argv[1]  # the number of state

	root = '/home/data/data/usera/xiatong/HMM/o_1hours_31day.npy'
	O = np.load(root)
	
	#init
	#Fixed parameters.
	N = 24 # the length of sequence for each iter(if the sequence is too long, then it's hard to train)
	M = N #The decode length each time
	K=int(in1) #The number of latent states.
	X=O[:,:2*24,:] #shape[R,N*n,L] the observation sequences for train
	D=O[:,:2*24,:] #shape[R,N2,L] the observation sequences for decode
	R=X.shape[0] #The number of sequences for train.
	L=X.shape[2] #The number of Gaussian components in each state.
	maxIter=5
	max_epoch=3
	print('==========o:',O.shape,'all data')
	print('==========X:',X.shape,'for training')
	print('==========D:',D.shape,'for decoding')
		
	#the model parameters
	pi=ones([R,K])*1.0/K # 2D version [R,K], for R regions, there are k states for each region
	A=ones([R,K])*1.0/K # need to be extend [R,K,K]
	sigma_=np.ones([K,L])*0.01 #fix 0.01
	mu_=np.random.random([K,L])*0.5+0.001

	#The latent variables
	alpha = zeros([R,N]) #need to be extend alpha[r][n][k]: for the n-th position of sequence r at state k.
	beta = zeros([R,N])  #need to be extend beta[r][n][k]: for the n-th position of sequence r at state k.
	gamma = zeros([R,N]) #need to be extend gamma[r][n][k]: probability that the n-th position of sequence r is at state k.
	xi = zeros([R,N-1]) #need to be extend xi[r][n][j][k]: the probability that the n-1-th position of sequence r is state j and the n-th position is k.	

	output='result_simpleHMM/result_R'+str(D.shape[0])+'_'+str(D.shape[1])+'_N'+str(N)+'_S'+str(K)+'max_iter'+str(maxIter)+'_pi_sigma/'
	print(output)
	if (os.path.exists(output)):
		pass
	else:
		os.mkdir(output)
	
	conf = SparkConf().setAppName("xiatong:HMM").setMaster('yarn')
	conf.set('spark.yarn.queue','usera')
	conf.set('spark.executor.memory','6g')
	conf.set('spark.driver.memory','6g')
	conf.set('spark.executor.instances','100')
	sc = SparkContext(conf=conf)
	sc.setLogLevel("ERROR")
	
	#print XRDD.getNumPartitions()
	XRDD = sc.parallelize(zeros([X.shape[0],X.shape[2]])).zipWithIndex().map(lambda x:extendx(x[1],x[0])) #for training 
	DRDD = sc.parallelize(zeros([D.shape[0],D.shape[2]])).zipWithIndex().map(lambda x:extendd(x[1],x[0])) #for decoding
	AlphaRDD = sc.parallelize(alpha).zipWithIndex().map(lambda x:extendabg(x[1],x[0]))
	BetaRDD = sc.parallelize(beta).zipWithIndex().map(lambda x:extendabg(x[1],x[0]))
	GammaRDD = sc.parallelize(gamma).zipWithIndex().map(lambda x:extendabg(x[1],x[0]))
	XiRDD = sc.parallelize(xi).zipWithIndex().map(lambda x:extendxi(x[1],x[0]))
	ARDD = sc.parallelize(A).zipWithIndex().map(lambda x:extendA(x[1],x[0]))
	PiRDD = sc.parallelize(pi).zipWithIndex().map(lambda x:(x[1],x[0]))
	AllRDD = XRDD.join(AlphaRDD).join(BetaRDD).join(GammaRDD).join(XiRDD).join(ARDD).join(PiRDD)
	AllRDD = AllRDD.map(lambda x: (x[0],(x[1][0][0][0][0][0][0],x[1][0][0][0][0][0][1],x[1][0][0][0][0][1],x[1][0][0][0][1],x[1][0][0][1],x[1][0][1],x[1][1])))
	print AllRDD.take(1)
	AllRDD = AllRDD.map(lambda x:extendmusigma(x))
	print AllRDD.take(1)
	for i in range(max_epoch):
		for j in range (maxIter):
			for n in range(0,X.shape[1],N):  # n is the beginning time slot
				if ( n + N - 1 < X.shape[1]): #the end is less than the last time slot
					print('epoch:',i,'itstrftimeer:',j,',begin:',n,',end:',n+N-1,'mu:', np.sum(mu_))
					print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'Estep:====================================')
					AllRDD = AllRDD.map(lambda x: calcAlpha(x,n)).map(lambda x: calcBeta(x,n)).map(lambda x: calcGamma(x,n)).map(lambda x: calcXi(x,n)).map(lambda x: updatePi(x,n)).map(lambda x: updateA(x,n))
					print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'Mstep:=====================================')
					AllRDD = AllRDD.map(lambda x:updateMuSigma(x,n))
			print AllRDD.take(1)
			print DRDD.take(1)
			decodeRDD = AllRDD.join(DRDD)
			print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'decode:=====================================')
			stateRDD = decodeRDD.map(lambda x: viterbi(x)).map(lambda x:(1,x)).groupByKey().map(lambda x: extraSate(x))
			state = zeros([R,X.shape[1]])
			state = stateRDD.collect()
			print state
			Pi = AllRDD.map(lambda x:x[1][6]).collect()
			A = AllRDD.map(lambda x:x[1][5]).collect()
			Mu = zeros([R,K,L])
			Mu = AllRDD.map(lambda x:x[1][7]).collect()
			save(output+'mu%d'%i+'_iter%d'%j,Mu)
			#savetxt(output+'sigma%d'%i+'_iter%d'%j,sigma,fmt='%.4f')
			savetxt(output+'pi%d'%i+'_iter%d'%j,Pi,fmt='%.4f')
			save(output+'A%d'%i+'_iter%d'%j,A)
			savetxt(output+'predict_stata%d'%i+'_iter%d'%j,state[0],fmt='%d')
		#re-random initialization 
		sigma=np.random.random((K,L))*0.5+0.001 
		mu=np.random.random([K,L])*0.5+0.001
