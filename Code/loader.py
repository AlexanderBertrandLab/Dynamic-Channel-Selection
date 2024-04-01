import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
import models

import scipy.io as sio
import time

import math
import os




def cross_subject_loader_HGD(subject,batch_size,train_split,path,shuffle=True,M=4):
	
	num_subjects = 14


	if(M==4):
		CHANNEL_SUBSET=np.asarray([202, 251,  75, 221])-1
	elif(M==8):
		CHANNEL_SUBSET=np.asarray([251, 248, 12, 23, 257, 47, 221, 202])-1
	elif(M==16):
		CHANNEL_SUBSET=np.asarray([188, 251, 52, 71, 257, 189, 160, 23, 51, 46, 248, 122, 31, 202, 63, 221])-1

	#Create dataset
	tr_ds=[]
	val_ds=[]
	test_ds=[]

	for k in range(num_subjects):

		if(k+1!=subject):
			#Load training data

			traindatapath = os.path.join(path,'train',str(k+1)+"traindatanode_3cm.npy")
			trainlabelpath = os.path.join(path,'train',str(k+1)+"trainlabelnode_3cm.npy")


			train_eeg_data = torch.Tensor(np.load(traindatapath))
			train_labels = torch.LongTensor(np.load(trainlabelpath))

			split = round(train_split*train_eeg_data.size(0))

			for i in range(train_eeg_data.size(0)):
				x = train_eeg_data[i,:,:]
				x=x.view(1,x.size(0),x.size(1))
				y = train_labels[i]
				if(i<=split):
					tr_ds.append([x,y])
				else:
					val_ds.append([x,y])

		else:

			#Load test data
			testdatapath = os.path.join(path,'test',str(k+1)+"testdatanode_3cm.npy")
			testlabelpath = os.path.join(path,'test',str(k+1)+"testlabelnode_3cm.npy")

			test_eeg_data = torch.Tensor(np.load(testdatapath))
			test_labels = torch.LongTensor(np.load(testlabelpath))

			for i in range(test_eeg_data.size(0)):
				x = test_eeg_data[i,:,:]
				x=x.view(1,x.size(0),x.size(1))
				y = test_labels[i]
				test_ds.append([x,y])

	trainloader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size,
										  shuffle=shuffle)
	valloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
										  shuffle=False)
	testloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
										  shuffle=False)
	return trainloader,valloader,testloader

def all_subject_loader_HGD(batch_size,train_split,path,shuffle=True,M=4):
	
	num_subjects = 14

	if(M==4):
		CHANNEL_SUBSET=np.asarray([202, 251,  75, 221])-1
	elif(M==8):
		CHANNEL_SUBSET=np.asarray([251, 248, 12, 23, 257, 47, 221, 202])-1
	elif(M==16):
		CHANNEL_SUBSET=np.asarray([188, 251, 52, 71, 257, 189, 160, 23, 51, 46, 248, 122, 31, 202, 63, 221])-1
	#Create dataset
	tr_ds=[]
	val_ds=[]
	test_ds=[]

	for k in range(num_subjects):
		#Load training data
		#traindatapath = path + str(k+1)+"traindata.npy"
		#trainlabelpath = path + str(k+1)+"trainlabel.npy"

		traindatapath = os.path.join(path,'train',str(k+1)+"traindatanode_3cm.npy")
		trainlabelpath = os.path.join(path,'train',str(k+1)+"trainlabelnode_3cm.npy")

		train_eeg_data = torch.Tensor(np.load(traindatapath))
		train_labels = torch.LongTensor(np.load(trainlabelpath))

		split = round(train_split*train_eeg_data.size(0))

		for i in range(train_eeg_data.size(0)):
			x = train_eeg_data[i,CHANNEL_SUBSET,:]
			x=x.view(1,x.size(0),x.size(1))
			y = train_labels[i]
			if(i<=split):
				tr_ds.append([x,y])
			else:
				val_ds.append([x,y])

		#Load test data
		testdatapath = os.path.join(path,'test',str(k+1)+"testdatanode_3cm.npy")
		testlabelpath = os.path.join(path,'test',str(k+1)+"testlabelnode_3cm.npy")

		test_eeg_data = torch.Tensor(np.load(testdatapath))
		test_labels = torch.LongTensor(np.load(testlabelpath))

		for i in range(test_eeg_data.size(0)):
			x = test_eeg_data[i,CHANNEL_SUBSET,:]
			x=x.view(1,x.size(0),x.size(1))
			y = test_labels[i]
			test_ds.append([x,y])

	trainloader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size,
										  shuffle=shuffle)
	valloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
										  shuffle=False)
	testloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
										  shuffle=False)
	return trainloader,valloader,testloader

def within_subject_loader_HGD(subject,batch_size,train_split,path,shuffle=True,M=4):

	if(M==4):
		CHANNEL_SUBSET=np.asarray([202, 251,  75, 221])-1
	elif(M==8):
		CHANNEL_SUBSET=np.asarray([251, 248, 12, 23, 257, 47, 221, 202])-1
	elif(M==16):
		CHANNEL_SUBSET=np.asarray([188, 251, 52, 71, 257, 189, 160, 23, 51, 46, 248, 122, 31, 202, 63, 221])-1


	traindatapath = os.path.join(path,'train',str(subject)+"traindatanode_3cm.npy")
	trainlabelpath = os.path.join(path,'train',str(subject)+"trainlabelnode_3cm.npy")

	train_eeg_data = torch.Tensor(np.load(traindatapath))
	train_labels = torch.LongTensor(np.load(trainlabelpath))
	split = round(train_split*train_eeg_data.size(0))

	tr_ds=[]
	val_ds = []
	split = round(train_split*train_eeg_data.size(0))
	for i in range(train_eeg_data.size(0)):
		x = train_eeg_data[i,CHANNEL_SUBSET,:]
		#x=x[::2,:]
		x=x.view(1,x.size(0),x.size(1))
		y = train_labels[i]
		if(i<= split):
			tr_ds.append([x,y])
		else:
			val_ds.append([x,y])

	testdatapath = os.path.join(path,'test',str(subject)+"testdatanode_3cm.npy")
	testlabelpath = os.path.join(path,'test',str(subject)+"testlabelnode_3cm.npy")

	test_eeg_data = torch.Tensor(np.load(testdatapath))
	test_labels = torch.LongTensor(np.load(testlabelpath))

	test_ds=[]
	for i in range(test_eeg_data.size(0)):
		x = test_eeg_data[i,CHANNEL_SUBSET,:]
		#x=x[::2,:]
		x=x.view(1,x.size(0),x.size(1))
		y = test_labels[i]
		test_ds.append([x,y])


	trainloader = torch.utils.data.DataLoader(tr_ds, batch_size=batch_size,
										  shuffle=shuffle)
	valloader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,
										  shuffle=False)
	testloader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,
										  shuffle=False)
	return trainloader,valloader,testloader