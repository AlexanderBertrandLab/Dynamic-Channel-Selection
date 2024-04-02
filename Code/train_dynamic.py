import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loader import all_subject_loader_HGD
from models import DynamicSelectionNet

from random import randint
from torchsummary import summary
from utils import json_write,apply_noise

parser = argparse.ArgumentParser(description='PyTorch Dynamic Channel Selection')

parser.add_argument('--M',type=int,default=4,

					help='number of channels')

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

parser.add_argument('--lamba', type=float, default=10.0, 

					help='sparsity loss weight')

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

	window_length=1125

	args = parser.parse_args()
	np.set_printoptions(linewidth=np.inf)

	cwd=os.path.dirname(__file__)
	dpath=os.path.dirname(cwd)

	#Paths for data, model and checkpoint
	data_path = os.path.join(dpath,'Data/')

	model_save_path = os.path.join(dpath,'Models','Dynamic'+args.name+'.pt')
	model_save_path_DSF = os.path.join(dpath,'Models','Dynamic'+args.name+'DSF.pt')
	model_save_path_distributed = os.path.join(dpath,'Models','Dynamic'+args.name+'distributed.pt')

	checkpoint_path = os.path.join(dpath,'Models','DynamicCheckpoint'+args.name+'.pt')

	performancepath = os.path.join(dpath,'Results',args.name+'.json')
	performancepath_DSF = os.path.join(dpath,'Results',args.name+'DSF.json')
	performancepath_distributed = os.path.join(dpath,'Results',args.name+'distributed.json')

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

	def supervised_loss(output,target):
		l = nn.CrossEntropyLoss()
		sup_loss = l(output,target)
		return sup_loss

	def reg_loss(model):
		reg = model.regularizer()
		return reg

	def sparsity_loss(selection):

		expected_selection=torch.mean(selection,dim=0)

		if(args.balanced):
			sparsity= torch.clamp(torch.max(expected_selection) - args.target,min=0.0)**2

		else:
			sparsity = torch.clamp(torch.mean(expected_selection) - args.target,min=0.0)**2
		
		return sparsity	
	

	def distributed_transfer_loss(scores_centralized,scores_distributed):
		l=nn.BCELoss()
		sup_loss=l(torch.sigmoid(scores_distributed),(scores_centralized>0).float())
		return sup_loss

	#train 1 epoch
	def train(train_loader, model, optimizer, epoch):

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

			output,sel = model(data)

			sup = supervised_loss(output,labels)
			reg = reg_loss(model)
			sparsity = sparsity_loss(sel)

			loss = sup + args.weight_decay*reg + args.lamba*sparsity				
			loss=loss/args.gradacc
			loss.backward()

			#Perform gradient accumulation
			if((i+1)%args.gradacc ==0):
				optimizer.step()
				optimizer.zero_grad()

			#running accuracy
			score, predicted = torch.max(output,1)
			total = predicted.size(0)
			correct = (predicted == labels).sum().item()
			running_acc = np.add(running_acc, np.array([correct,total]))

			# print statistics
			running_loss += loss.item()
			running_reg += args.weight_decay*reg.item()
			running_sup_loss += sup.item()
			running_sparsity_loss += args.lamba*sparsity.item()

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

	def validate(val_loader,model,epoch):

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

				output,sel = model(data)

				sup = supervised_loss(output,labels)
				reg = reg_loss(model)
				sparsity = sparsity_loss(sel)

				loss = sup + args.weight_decay*reg + args.lamba*sparsity		

				#running accuracy
				score, predicted = torch.max(output,1)
				total = predicted.size(0)
				correct = (predicted == labels).sum().item()
				val_acc = np.add(val_acc, np.array([correct,total]))
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

			#Training set
			total = 0
			correct = 0
			total_rate=[]

			for i, (data, labels) in enumerate(train_loader):

				data=apply_noise(data,args.noise_prob)

				if(enable_cuda):
					data= data.cuda()
					labels = labels.cuda()

				output,sel = model(data)
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

			#Validation set
			total = 0
			correct = 0
			total_rate=[]

			for i, (data, labels) in enumerate(val_loader):

				data=apply_noise(data,args.noise_prob)

				if(enable_cuda):
					data= data.cuda()
					labels = labels.cuda()

				output,sel = model(data)
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

			#Test set
			total = 0
			correct = 0
			total_rate=[]

			for i, (data, labels) in enumerate(test_loader):

				data=apply_noise(data,args.noise_prob)

				if(enable_cuda):
					data= data.cuda()
					labels = labels.cuda()

				output,sel = model(data)
				score, predicted = torch.max(output,1)
				total += predicted.size(0)
				correct += (predicted == labels).sum().item()

				expected_selection=torch.mean(sel,dim=0)
				total_rate.append(expected_selection.cpu().data)

			test_acc = correct/total

			test_rate=torch.mean(torch.stack(total_rate,dim=0),dim=0)

			if(args.verbose):
				print('Test set accuracy: %d %%' % (100 * test_acc))
				print('Test set rate: ' + str(100 * torch.squeeze(test_rate).cpu().data.numpy().T))
			return tr_acc,val_acc,test_acc,test_rate



	def train_distributed(train_loader, model, model_centralized, optimizer, epoch):

		model_centralized.eval()
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

			scores_centralized = torch.squeeze(model_centralized.selection_layer.channel_scorer(data),-1)
			scores_distributed = torch.squeeze(model.selection_layer.channel_scorer(data),-1)

			output,sel = model(data)

			sup = distributed_transfer_loss(scores_centralized,scores_distributed)
			reg = reg_loss(model)
			sparsity = sparsity_loss(sel)

			loss = sup			
			
			loss=loss/args.gradacc
			loss.backward()

			#Perform gradient accumulation
			if((i+1)%args.gradacc ==0):
				optimizer.step()
				optimizer.zero_grad()

			#running accuracy
			score, predicted = torch.max(output,1)
			total = predicted.size(0)
			correct = (predicted == labels).sum().item()
			running_acc = np.add(running_acc, np.array([correct,total]))

			# print statistics
			running_loss += loss.item()
			running_reg += args.weight_decay*reg.item()
			running_sup_loss += sup.item()
			running_sparsity_loss += args.lamba*sparsity.item()

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

	#Load data
	num_subjects = 14

	input_dim=[args.M,window_length]
	train_loader,val_loader,test_loader = all_subject_loader_HGD(batch_size=args.batch_size,train_split=args.train_split,path=data_path,M=args.M)

# ################################################################ Centralized Training #################################################################################

	if(args.verbose):
		print('Start Centralized Training')

	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	model = DynamicSelectionNet(input_dim,distributionmode='Centralized',enable_DSF=False)

	pretrained_model_path=os.path.join(dpath,'Models','Pretrained_M'+str(args.M)+'.pt')
	model.network.load_state_dict(torch.load(pretrained_model_path))

	if(enable_cuda):
		model.cuda()

	summary(model,input_size=(1,args.M,window_length),depth=5)

	optimizer = torch.optim.Adam([{'params': model.selection_layer.parameters()},
								{'params': model.network.parameters(), 'lr': args.lr_finetune}], lr=args.lr)

	prev_val_loss = 100
	patience_timer = 0
	early_stop = False
	epoch = 0

	while epoch in range(args.epochs) and (not early_stop):

		#Perform training step
		train(train_loader, model,optimizer,epoch)
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
		print('Centralized Training finished')

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

################################################################ ADD DSF #################################################################################

	if(args.verbose):
		print('Start DSF Training')

	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	pretrained_model = DynamicSelectionNet(input_dim,distributionmode='Centralized',enable_DSF=False)
	pretrained_model.load_state_dict(torch.load(model_save_path))

	model = DynamicSelectionNet(input_dim,distributionmode='Centralized',enable_DSF=args.enable_DSF)
	if(enable_cuda):
		model.cuda()
		pretrained_model.cuda()

	model.network.load_state_dict(pretrained_model.network.state_dict())
	model.selection_layer.channel_scorer.load_state_dict(pretrained_model.selection_layer.channel_scorer.state_dict())

	summary(model,input_size=(1,args.M,window_length),depth=5)

	if(args.enable_DSF):
		optimizer = torch.optim.Adam([{'params': model.selection_layer.DSF.parameters()},
									{'params': model.selection_layer.channel_scorer.parameters(), 'lr': args.lr_finetune},
									{'params': model.network.parameters(), 'lr': args.lr_finetune}], lr=args.lr)
	else:
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)	

	prev_val_loss = 100
	patience_timer = 0
	early_stop = False
	epoch = 0

	while epoch in range(args.epochs) and (not early_stop):

		#Perform training step
		train(train_loader, model,optimizer,epoch)
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
		print('DSF Training finished')

	#Store subject independent model
	model.load_state_dict(torch.load(checkpoint_path))
	torch.save(model.state_dict(), model_save_path_DSF)

	#Evaluate model
	tr_acc,val_acc,test_acc,rate=test(train_loader,val_loader,test_loader,model,epoch)

	full_dict=vars(args)

	results_dict={
		"Test accuracy": test_acc,
		"Rate": torch.squeeze(rate).cpu().data.numpy().round(3).T.tolist(),
	}

	full_dict.update(results_dict)
	json_write(performancepath_DSF,full_dict)



################################################################ Distributed Step 1 #################################################################################

	if(args.verbose):
		print('Start training')

	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	centralized_model = DynamicSelectionNet(input_dim,distributionmode='Centralized',enable_DSF=args.enable_DSF)
	centralized_model.load_state_dict(torch.load(model_save_path_DSF))

	model = DynamicSelectionNet(input_dim,distributionmode=args.distributionmode,enable_DSF=args.enable_DSF)
	model.network.load_state_dict(centralized_model.network.state_dict())
	if(args.enable_DSF):
		model.selection_layer.DSF.load_state_dict(centralized_model.selection_layer.DSF.state_dict())

	if(enable_cuda):
		model.cuda()
		centralized_model.cuda()

	summary(model,input_size=(1,args.M,window_length),depth=5)

	if(args.enable_DSF):
		optimizer = torch.optim.Adam([{'params': model.selection_layer.channel_scorer.parameters()},
									{'params': model.selection_layer.DSF.parameters(), 'lr': 0.0},
									{'params': model.network.parameters(), 'lr': args.lr_finetune}], lr=args.lr)
	else:
		optimizer = torch.optim.Adam([{'params': model.selection_layer.channel_scorer.parameters()},
									{'params': model.network.parameters(), 'lr': args.lr_finetune}], lr=args.lr)	

	prev_val_loss = 100
	patience_timer = 0
	early_stop = False
	epoch = 0

	while epoch in range(args.epochs) and (not early_stop):

		#Perform training step
		train_distributed(train_loader, model,centralized_model,optimizer,epoch)
		val_loss = validate(val_loader,model,epoch)
		tr_acc,val_acc,test_acc,rate=test(train_loader,val_loader,test_loader,model,epoch)

		if((epoch >50) and (val_loss>prev_val_loss-args.stop_delta)):
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
		print('Distributed Step 1 finished')

	#Store subject independent model
	model.load_state_dict(torch.load(checkpoint_path))
	torch.save(model.state_dict(), model_save_path_distributed)

	#Evaluate model
	tr_acc,val_acc,test_acc,rate=test(train_loader,val_loader,test_loader,model,epoch)

################################################################ Distributed Step 2 #################################################################################

	if(args.verbose):
		print('Start training')

	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	model = DynamicSelectionNet(input_dim,distributionmode=args.distributionmode,enable_DSF=args.enable_DSF)
	model.load_state_dict(torch.load(model_save_path_distributed))

	if(enable_cuda):
		model.cuda()

	summary(model,input_size=(1,args.M,window_length),depth=5)

	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_finetune)

	prev_val_loss = 100
	patience_timer = 0
	early_stop = False
	epoch = 0

	while epoch in range(args.epochs) and (not early_stop):

		#Perform training step
		train(train_loader, model,optimizer,epoch)
		val_loss = validate(val_loader,model,epoch)
		tr_acc,val_acc,test_acc,rate=test(train_loader,val_loader,test_loader,model,epoch)

		if((epoch >50) and (val_loss>prev_val_loss-args.stop_delta)):
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
		print('Distributed Step 2 finished')

	#Store subject independent model
	model.load_state_dict(torch.load(checkpoint_path))
	torch.save(model.state_dict(), model_save_path_distributed)

	#Evaluate model
	tr_acc,val_acc,test_acc,rate=test(train_loader,val_loader,test_loader,model,epoch)

	full_dict=vars(args)

	results_dict={
		"Test accuracy": test_acc,
		"Rate": torch.squeeze(rate).cpu().data.numpy().round(3).T.tolist(),
	}

	full_dict.update(results_dict)
	json_write(performancepath_distributed,full_dict)
if __name__ == '__main__':

	main()
