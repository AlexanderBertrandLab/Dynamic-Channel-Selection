import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
from loader import within_subject_loader_HGD, all_subject_loader_HGD
from models import *

import statistics
from random import randint
from torchsummary import summary
import math
import json
from utils import json_write,exponential_decay_schedule,apply_noise

parser = argparse.ArgumentParser(description='PyTorch Greedy Conditional Mutual Information')

parser.add_argument('--M',type=int,default=4,

					help='number of channels')

parser.add_argument('--K',type=int,default=2,

					help='number of CMI iterations')

parser.add_argument('--epochs',type=int, default=100,

					help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', type=int, default=64, 

					help='mini-batch size')

parser.add_argument('--gradacc', type = int, default=1,

					help='gradient accumulation')

parser.add_argument('--lr', '--learning-rate', type=float, default=0.001,

					 help='learning rate')

parser.add_argument('--lr_finetune', type=float, default=0.0001,

					help='learning rate during finetuning')

parser.add_argument('--weight-decay', '--wd', type=float, default=5e-4, 

					help='weight decay')

parser.add_argument('--lamba', type=float, default=0.1, 

					help='regularization weight')

parser.add_argument('--train_split',type=float,default=0.8,

					help='training-validation data split')

parser.add_argument('--patience', type=int, default=10,

					help='amount of epochs before early stopping')

parser.add_argument('--stop_delta', type=float, default=1e-3,

					help='maximal drop in validation loss for early stopping')

parser.add_argument('--seed',type=int,default=0,

					help='random seed, 0 indicates randomly chosen seed')

parser.add_argument('--name', type=str,default="Default",

					help = 'experiment name')

parser.add_argument('--target', type=float, default=0.5,

					help='Target transmission rate')

parser.add_argument('--distributionmode', type=str, default='Centralized',

					help='Mode of operation for the channel scoring function, options are "Centralized", "Distributed", "Distributed-Feedback" ')

parser.add_argument('--noise_prob', type=float, default=0.0,

					help = 'Probability of noise burst occuring on each channel')

parser.add_argument('-enable_DSF', action="store_true", default=False,dest="enable_DSF",

					help = 'Include DSF module in network')

parser.add_argument('-balanced', action="store_true", default=False,dest="balanced",

					help = 'Enforce balanced transmission load')

parser.add_argument('-v', action="store_true", default=True, dest="verbose")

def main():

	global args,enable_cuda
################################################################ INIT #################################################################################
	
	args = parser.parse_args()
	np.set_printoptions(linewidth=np.inf)

	cwd=os.getcwd()
	dpath=os.path.dirname(cwd)	

	#Paths for data, model and checkpoint
	data_path = os.path.join(dpath,'Data/')
	
	dpath=os.path.dirname(cwd)

	model_save_path = os.path.join(dpath,'Models','CMI'+args.name+'.pt')
	checkpoint_path = os.path.join(dpath,'Models','CMI'+args.name+'.pt')

	performancepath = os.path.join(dpath,'Results',args.name+'.json')

	if not os.path.isdir(os.path.join(dpath,'Models')):
		os.makedirs(os.path.join(dpath,'Models'))

	if not os.path.isdir(os.path.join(dpath,'Results')):
		os.makedirs(os.path.join(dpath,'Results'))

	#Check if CUDA is available
	enable_cuda = torch.cuda.is_available()

	#Set random seed
	if(args.seed==0):
		args.seed=randint(1,99999)

	#Initialize devices with random seed
	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	training_accs = []
	val_accs=[]
	test_accs = []



	def supervised_loss_finetune(output,target):
		l = nn.CrossEntropyLoss()
		sup_loss = l(output,target)
		return sup_loss


	def supervised_loss(alloutput,target):
		CE = nn.CrossEntropyLoss(reduction='mean')
		l=0.0
		for k in range(args.K):
			lk = CE(alloutput[k,:,:],target)
			l +=lk
		sup_loss = l/args.K
		return sup_loss

	def reg_loss(model):
		reg=model.regularizer()
		return reg

	def sparsity_loss(selection):

		expected_selection=torch.mean(selection,dim=0)

		if(args.balanced):
			sparsity= torch.clamp(torch.max(expected_selection) - args.target,min=0.0)**2

		else:
			sparsity = torch.clamp(torch.mean(expected_selection) - args.target,min=0.0)**2
		
		return sparsity	



	#train 1 epoch
	def train(train_loader, model, optimizer, epoch,finetune=False):

		model.train()

		for i, (data, labels) in enumerate(train_loader):

			data = apply_noise(data,args.noise_prob)

			if(enable_cuda):
				data= data.cuda()
				labels = labels.cuda()

			if(i==0):
				running_loss = 0.0
				running_reg = 0.0
				running_sup_loss = 0.0
				running_sparsity_loss=0.0
				running_acc = np.array([0,0])
				running_rate=[]

			output,alloutput,sel = model(data)

			if(not finetune):
				sup = supervised_loss(alloutput,labels)
			else:
				sup = supervised_loss_finetune(output,labels)

			reg = reg_loss(model)
			sparsity = sparsity_loss(sel)

			loss = sup + args.weight_decay*reg + args.lamba*sparsity				
			loss=loss/args.gradacc
			loss.backward()

			#Perform gradient accumulation
			if((i+1)%args.gradacc ==0):
				optimizer.step()
				# print(model.selector.logitpredictor.conv1.weight[0,:])
				optimizer.zero_grad()

			#running accuracy
			score, predicted = torch.max(output,1)
			total = predicted.size(0)
			correct = (predicted == labels).sum().item()
			running_acc = np.add(running_acc, np.array([correct,total]))

			# print statistics
			running_loss += loss.item()
			running_reg += reg.item()
			running_sup_loss += sup.item()
			running_sparsity_loss += sparsity.item()

			expected_selection=torch.mean(sel,dim=0)
			running_rate.append(expected_selection.detach())

			N = len(train_loader)
			if(i==N-1):
				running_rate=torch.mean(torch.squeeze(torch.stack(running_rate,dim=0)),dim=0)

				if(args.verbose):
					print('[%d, %5d] loss: %.3f acc: %d %% supervised loss: %.3f regularization loss %.3f sparsity loss %.5f'%
							(epoch + 1, i + 1, running_loss / N, 100*running_acc[0]/running_acc[1], running_sup_loss/N, running_reg/N, running_sparsity_loss/N))
					print('Rate: ' + str(100*torch.squeeze(running_rate).cpu().data.numpy().T) )


				running_loss = 0.0
				running_reg = 0.0
				running_sup_loss = 0.0
				running_sparsity_loss=0.0
				running_acc = (0,0)
				running_rate=[]

	def validate(val_loader,model,epoch,finetune=False):
		with torch.no_grad():
			model.eval()

			for i, (data, labels) in enumerate(val_loader):

				data = apply_noise(data,args.noise_prob)

				if(enable_cuda):
					data= data.cuda()
					labels = labels.cuda()

				if(i==0):
					val_loss = 0.0
					val_acc = np.array([0,0])

				output,alloutput,sel = model(data)

				if(not finetune):
					sup = supervised_loss(alloutput,labels)
				else:
					sup = supervised_loss_finetune(output,labels)
				reg = reg_loss(model)
				sparsity = sparsity_loss(sel)

				loss = sup + args.weight_decay*reg + args.lamba*sparsity	

				#running accuracy
				score, predicted = torch.max(output,1)
				total = predicted.size(0)
				correct = (predicted == labels).sum().item()
				val_acc = np.add(val_acc, np.array([correct,total]))

				# print statistics
				val_loss += loss.item()
				N = len(val_loader)
				if(i == N-1):
					if(args.verbose):
						print('[%d, %5d] Validation loss: %.3f Validation accuracy: %d %%'%
							(epoch + 1, i + 1, val_loss / N,100*val_acc[0]/val_acc[1] ))

		return val_loss/N

	def test(train_loader,val_loader,test_loader, model,epoch=0):

		with torch.no_grad():
			model.eval()

			total = 0
			correct = 0
			total_rate=[]

			for i, (data, labels) in enumerate(train_loader):

				data=apply_noise(data,args.noise_prob)

				if(enable_cuda):
					data= data.cuda()
					labels = labels.cuda()

				output,alloutput,sel = model(data)
				score, predicted = torch.max(output,1)
				total += predicted.size(0)
				correct += (predicted == labels).sum().item()

				expected_selection=torch.mean(sel,dim=0)
				total_rate.append(expected_selection.cpu().data)

			tr_acc = correct/total

			rate=torch.mean(torch.stack(total_rate,dim=0),dim=0)

			if(args.verbose):
				print('Training set accuracy: %d %%' % (100 * tr_acc))
				print('Training set rate: ' + str(100 * torch.squeeze(rate).cpu().data.numpy().T))

			model.eval()

			total = 0
			correct = 0
			total_rate=[]

			for i, (data, labels) in enumerate(val_loader):

				data=apply_noise(data,args.noise_prob)

				if(enable_cuda):
					data= data.cuda()
					labels = labels.cuda()

				output,alloutput,sel = model(data)
				score, predicted = torch.max(output,1)
				total += predicted.size(0)
				correct += (predicted == labels).sum().item()

				expected_selection=torch.mean(sel,dim=0)
				total_rate.append(expected_selection.cpu().data)

			val_acc = correct/total

			rate=torch.mean(torch.stack(total_rate,dim=0),dim=0)

			if(args.verbose):
				print('Validation set accuracy: %d %%' % (100 * val_acc))
				print('Validation set rate: ' + str(100 * torch.squeeze(rate).cpu().data.numpy().T))

			total = 0
			correct = 0
			total_rate=[]

			for i, (data, labels) in enumerate(test_loader):

				data=apply_noise(data,args.noise_prob)

				if(enable_cuda):
					data= data.cuda()
					labels = labels.cuda()

				output,alloutput,sel = model(data)
				score, predicted = torch.max(output,1)
				total += predicted.size(0)
				correct += (predicted == labels).sum().item()

				expected_selection=torch.mean(sel,dim=0)
				total_rate.append(expected_selection.cpu().data)

			test_acc = correct/total

			rate=torch.mean(torch.stack(total_rate,dim=0),dim=0)

			if(args.verbose):
				print('Test set accuracy: %d %%' % (100 * test_acc))
				print('Test set rate: ' + str(100 * torch.squeeze(rate).cpu().data.numpy().T))
			return tr_acc,val_acc,test_acc,rate

	#Load data
	num_subjects = 14

	input_dim=[args.M,1125]
	train_loader,val_loader,test_loader = all_subject_loader_HGD(batch_size=args.batch_size,train_split=args.train_split,path=data_path,M=args.M)


	################################################################ SUBJECT-INDEPENDENT CHANNEL SELECTION #################################################################################

	if(args.verbose):
		print('Start training')

	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	temperature_schedule = exponential_decay_schedule(5.0,1.0,args.epochs,int(args.epochs*3/4))

	model = CMI(input_dim,enable_DSF=args.enable_DSF,K=args.K,batchnorm=False)

	if(enable_cuda):
		model.cuda()
	summary(model,input_size=(1,args.M,1125),depth=5)


	pretrained_model_path=os.path.join(dpath,'Models','Pretrained_M'+str(args.M)+'.pt')
	if(enable_cuda):
		model.predictor.load_state_dict(torch.load(pretrained_model_path))
	else:
		model.predictor.load_state_dict(torch.load(pretrained_model_path,map_location='cpu'))

	if(args.enable_DSF):
		optimizer = torch.optim.Adam([{'params': model.selector.parameters()},
									{'params': model.DSF.parameters()},
									{'params': model.predictor.parameters(), 'lr': args.lr_finetune}],lr=args.lr)
	else:
		optimizer = torch.optim.Adam([{'params': model.selector.parameters()},
									{'params': model.predictor.parameters(), 'lr': args.lr_finetune}],lr=args.lr)		

	prev_val_loss = 100
	patience_timer = 0
	early_stop = False
	epoch = 0

	while epoch in range(args.epochs) and (not early_stop):

		model.set_temperature(temperature_schedule[epoch])

		#Perform training step
		train(train_loader, model, optimizer,epoch)
		val_loss = validate(val_loader,model,epoch)
		tr_acc,val_acc,test_acc,rate=test(train_loader,val_loader,test_loader,model,epoch)

		if((epoch >100) and (val_loss>prev_val_loss-args.stop_delta)):
			patience_timer+=1
			if(args.verbose):
				print('Early stopping timer ', patience_timer)
			if(patience_timer == args.patience):
				early_stop = True
		else:
			patience_timer=0
			torch.save(model.state_dict(),checkpoint_path)
			prev_val_loss = val_loss

		epoch+=1

	if(args.verbose):
		print('Channel selection finished')

	#Store subject independent model
	model.load_state_dict(torch.load(checkpoint_path))
	torch.save(model.state_dict(), model_save_path)

	tr_acc,val_acc,test_acc,rate=test(train_loader,val_loader,test_loader,model,epoch)

################################################################ SUBJECT-INDEPENDENT CHANNEL SELECTION #################################################################################
################################################################ SUBJECT-INDEPENDENT CHANNEL SELECTION #################################################################################
################################################################ SUBJECT-INDEPENDENT CHANNEL SELECTION #################################################################################
################################################################ SUBJECT-INDEPENDENT CHANNEL SELECTION #################################################################################
################################################################ SUBJECT-INDEPENDENT CHANNEL SELECTION #################################################################################

	if(args.verbose):
		print('Start finetune')

	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


	summary(model,input_size=(1,args.M,1125),depth=5)

	model = CMI(input_dim,enable_DSF=args.enable_DSF,K=args.K,batchnorm=False,single_loop=True)
	model.load_state_dict(torch.load(model_save_path))
	if(enable_cuda):
		model.cuda()

	optimizer = torch.optim.Adam([{'params': model.selector.parameters()},
								{'params': model.predictor.parameters(), 'lr': args.lr_finetune}],lr=0.0)


	if(args.enable_DSF):
		optimizer = torch.optim.Adam([{'params': model.selector.parameters()},
									{'params': model.softfuser.parameters(), 'lr': args.lr_finetune},
									{'params': model.predictor.parameters(), 'lr': args.lr_finetune}],lr=0.0)
	else:
		optimizer = torch.optim.Adam([{'params': model.selector.parameters()},
									{'params': model.predictor.parameters(), 'lr': args.lr_finetune}],lr=0.0)		

	prev_val_loss = 100
	patience_timer = 0
	early_stop = False
	epoch = 0

	model.set_freeze(True)				

	while epoch in range(args.epochs) and (not early_stop):

		model.set_temperature(temperature_schedule[-1])

		#Perform training step
		train(train_loader, model, optimizer,epoch,finetune=True)
		val_loss = validate(val_loader,model,epoch,finetune=True)
		tr_acc,val_acc,test_acc,rate=test(train_loader,val_loader,test_loader,model,epoch)

		if((epoch >10) and (val_loss>prev_val_loss-args.stop_delta)):
			patience_timer+=1
			if(args.verbose):
				print('Early stopping timer ', patience_timer)
			if(patience_timer == args.patience):
				early_stop = True
		else:
			patience_timer=0
			torch.save(model.state_dict(),checkpoint_path)
			prev_val_loss = val_loss

		epoch+=1

	if(args.verbose):
		print('Channel selection finished')

	#Store subject independent model
	model.load_state_dict(torch.load(checkpoint_path))
	torch.save(model.state_dict(), model_save_path)

	#Evaluate model
	tr_acc,val_acc,test_acc,rate=test(train_loader,val_loader,test_loader,model,epoch)

	full_dict=vars(args)

	results_dict={
		"Test accuracy": test_acc,
		"Rate": torch.squeeze(rate).cpu().data.numpy().round(3).T.tolist(),
	}

	full_dict.update(results_dict)

	json_write(performancepath,full_dict)

if __name__ == '__main__':

	main()
