import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch.nn.parameter import Parameter


epsilon = 1e-10

def init_weights(m):
	if (type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d):
		torch.nn.init.xavier_uniform_(m.weight)


def set_cuda(x):
	if(torch.cuda.is_available()):
		return x.cuda()
	else:
		return x



def batch_cov(points):
	points=points.permute(0,2,1)
	B, N, D = points.size()
	mean = points.mean(dim=1).unsqueeze(1)
	diffs = (points - mean).reshape(B * N, D)
	prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
	bcov = prods.sum(dim=1) / (N - 1)
	return bcov  # (B, D, D)



class MSFBCNN(nn.Module):
	def __init__(self,input_dim,output_dim=4,batchnorm=True):
		super(MSFBCNN, self).__init__()
		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

		self.T=input_dim[1]
		self.FT=10
		self.FS=10
		self.C=input_dim[0]
		self.output_dim = output_dim
		
		# Parallel temporal convolutions
		self.conv1a = nn.Conv2d(1, self.FT, (1, 65), padding = (0,32),bias=False)
		self.conv1b = nn.Conv2d(1, self.FT, (1, 41), padding = (0,20),bias=False)
		self.conv1c = nn.Conv2d(1, self.FT, (1, 27), padding = (0,13),bias=False)
		self.conv1d = nn.Conv2d(1, self.FT, (1, 17), padding = (0,8),bias=False)

		self.batchnorm1 = nn.BatchNorm2d(4*self.FT, False)
		
		# Spatial convolution
		self.conv2 = nn.Conv2d(4*self.FT, self.FS, (self.C,1),padding=(0,0),groups=1,bias=False)
		self.batchnorm2 = nn.BatchNorm2d(self.FS, False)

		#Temporal average pooling
		self.pooling2 = nn.AvgPool2d(kernel_size=(1, 75),stride=(1,15),padding=(0,0))

		self.drop=nn.Dropout(0.5)

		#Classification
		self.fc1 = nn.Linear(self.FS*(1+math.floor((self.T-75)/15)), self.output_dim)

		self.batchnorm=batchnorm

		self.layers=self.create_layers_field()
		self.apply(init_weights)

	def forward(self, x):

		# Layer 1
		x1 = self.conv1a(x)
		x2 = self.conv1b(x)
		x3 = self.conv1c(x)
		x4 = self.conv1d(x)

		x = torch.cat([x1,x2,x3,x4],dim=1)
		if(self.batchnorm):
			x = self.batchnorm1(x)

		# Layer 2
		if(self.batchnorm):
			x = torch.pow(self.batchnorm2(self.conv2(x)),2)
		else:
			x = torch.pow(self.conv2(x),2)
		x = self.pooling2(x)
		x = torch.log(x+1e-16)
		x = self.drop(x)
		
		# FC Layer
		x = torch.flatten(x,start_dim=1)
		x = self.fc1(x)
		return x

	def regularizer(self):
		#L2-Regularization of layers
		reg=self.floatTensor([0])
		for i,layer in enumerate(self.layers):
			if(type(layer) == nn.Conv2d or type(layer) == nn.Linear):
				reg+=torch.sum(torch.pow(layer.weight,2))
		return reg

	def create_layers_field(self):
		layers = []
		for idx, m in enumerate(self.modules()):
			if(type(m) == nn.Conv2d or type(m) == nn.Linear):
				layers.append(m)
		return layers


class Gumbel(nn.Module):
	def __init__(self, eps=1e-8):
		super(Gumbel, self).__init__()
		self.eps = eps

	def forward(self, x, gumbel_temp=1.0, gumbel_noise=True):
	
		gumbel_hard_temp=0.1

		if not self.training:  # no Gumbel noise during inference
			return (x >= 0).float(),(x >= 0).float()

		rate=torch.sigmoid(x / gumbel_hard_temp)

		if gumbel_noise:
			eps = self.eps
			U1, U2 = torch.rand_like(x), torch.rand_like(x)
			g1, g2 = -torch.log(-torch.log(U1 + eps)+eps), - \
                torch.log(-torch.log(U2 + eps)+eps)
			x = x + g1 - g2

		soft = torch.sigmoid(x / gumbel_temp)
		hard = ((soft >= 0.5).float() - soft).detach() + soft

		assert not torch.any(torch.isnan(hard))
		return hard,rate




class DSF(nn.Module):
	def __init__(self,M,T):
		super(DSF, self).__init__()

		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
		self.M=M
		self.T=T
		FN=50
		self.mlp=nn.Sequential(
			nn.Linear(self.M**2,FN),
			nn.ReLU(),
			nn.Linear(FN,self.M**2+self.M)
		)

	def forward(self, x):
		#Prevent covariance matrix of becoming degenerate
		x=x+1e-16*set_cuda(torch.randn(x.size()))
		C=batch_cov(x.squeeze(1))
		C=torch.flatten(torch.triu(C),start_dim=1)
		wandb=self.mlp(C)

		W=wandb[:,:self.M**2]
		b=wandb[:,self.M**2:]

		W=W.view(-1,1,self.M,self.M)
		b=b.view(-1,1,self.M,1)

		out=torch.matmul(x.permute(0,1,3,2),W).permute(0,1,3,2)
		out=out + b

		out=out+x


		return out



class ChannelScorer_Centralized(nn.Module):
	def __init__(self, M,T):

		super(ChannelScorer_Centralized, self).__init__()
		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

		self.M=M
		FT=10
		FS=10

		self.conv1=nn.Conv2d(1,FT,kernel_size=(1,17),padding=(0,8))
		self.batchnorm1 = nn.BatchNorm2d(FT, False)

		self.conv2=nn.Conv2d(FT,FS,kernel_size=(M,1),padding=(0,0))
		self.pooling2 = nn.AvgPool2d(kernel_size=(1, 75),stride=(1,15),padding=(0,0))

		self.batchnorm2 = nn.BatchNorm2d(FS, False)

		#Temporal average pooling
		self.pooling2 = nn.AvgPool2d(kernel_size=(1, 75),stride=(1,15),padding=(0,0))

		self.fc1 = nn.Linear(int(FS*(1+math.floor(T-75)/15)), int(M),bias=True)


	def forward(self, x):

		x=self.batchnorm1(self.conv1(x))

		x = torch.pow(self.batchnorm2(self.conv2(x)),2)
		x = self.pooling2(x)

		x = torch.log(torch.clamp(x,min=epsilon))
		
		# FC Layer
		x = torch.flatten(x,start_dim=1)
		x = self.fc1(x)

		out=x.view(-1,self.M,1)

		return out


class ChannelScorer_Distributed(nn.Module):
	def __init__(self,M,T):

		super(ChannelScorer_Distributed, self).__init__()
		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

		self.M=M
		FT=10
		FS=10

		self.conv1=nn.Conv2d(1,FT,kernel_size=(1,17),padding=(0,8))

		#Temporal average pooling
		self.pooling2 = nn.AvgPool2d(kernel_size=(1, 75),stride=(1,15),padding=(0,0))


		self.linears=nn.ModuleList()
		for i in range(M):
			self.linears.append(nn.Linear(int(FS*(1+math.floor(T-75)/15)), int(1),bias=True))

	def forward(self, x):

		out=[]
		for i in range(self.M):
			y=self.conv1(x[:,:,[i],:])
			y=torch.pow(y,2)
			y=self.pooling2(y)
			y=torch.log(torch.clamp(y,min=epsilon))
			y=torch.flatten(y,start_dim=1)
			y=self.linears[i](y)
			out.append(y)

		out=torch.cat(out,dim=1)
		out=out.view(-1,self.M,1)


		return out



class ChannelScorer_Distributed_Feedback(nn.Module):
	def __init__(self,M,T):

		super(ChannelScorer_Distributed_Feedback, self).__init__()
		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

		self.M=M
		self.FT=10
		self.FS=10
		self.FD=10

		self.conv1=nn.Conv2d(1,self.FT,kernel_size=(1,17),padding=(0,8))

		#Temporal average pooling
		self.pooling2 = nn.AvgPool2d(kernel_size=(1, 75),stride=(1,15),padding=(0,0))

		self.fc1 = nn.Linear(int(self.FS*(1+math.floor(T-75)/15)), int(self.FD),bias=True)

		FN=50
		self.fc2=nn.Linear(M*self.FD,FN)
		self.fc3=nn.Linear(FN,M)

	def forward(self, x):

		out=[]
		for i in range(self.M):
			y=self.conv1(x[:,:,[i],:])
			y=torch.pow(y,2)
			y=self.pooling2(y)
			y=torch.log(torch.clamp(y,min=epsilon))
			y=torch.flatten(y,start_dim=1)
			y=self.fc1(y)
			out.append(y)
		
		out=torch.cat(out,dim=1)
		out=self.fc3(F.relu(self.fc2(out)))

		out=out.view(-1,self.M,1)


		return out





class DynamicSelectionNet(nn.Module):
	
	def __init__(self,input_dim,distributionmode='Centralized',enable_DSF=False):
		super(DynamicSelectionNet,self).__init__()
		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

		self.M = input_dim[0]
		self.T = input_dim[1]
		self.input_dim = input_dim
		self.output_dim = 4
			
		self.network = MSFBCNN(input_dim=[self.M,self.T],output_dim=self.output_dim)
		self.selection_layer = DynamicSelectionLayer(self.M,self.T,distributionmode=distributionmode,enable_DSF=enable_DSF)

		self.layers = self.create_layers_field()
		self.apply(init_weights)
		
	def forward(self,x):

		y_selected,selection = self.selection_layer(x)
		out = self.network(y_selected)

		return out,selection

	def regularizer(self):
		#L2-Regularization of other layers
		reg=self.floatTensor([0])
		for i,layer in enumerate(self.layers):
				if(type(layer) == nn.Conv2d or type(layer) == nn.Linear):
					reg+=torch.sum(torch.pow(layer.weight,2))
		return reg

	def create_layers_field(self):
		layers = []
		for idx, m in enumerate(self.modules()):
			if(type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == DynamicSelectionLayer):
				layers.append(m)
		return layers

class DynamicSelectionLayer(nn.Module):
	def __init__(self,M,T,distributionmode='Centralized',enable_DSF=False):

		super(DynamicSelectionLayer, self).__init__()
		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
		self.M = M

		if(distributionmode.lower()=='centralized'):
			self.channel_scorer=ChannelScorer_Centralized(M,T)
		elif(distributionmode.lower()=='distributed'):
			self.channel_scorer=ChannelScorer_Distributed(M,T)
		else:
			self.channel_scorer=ChannelScorer_Distributed_Feedback(M,T)

		self.gumbel=Gumbel()
		self.freeze=False

		self.enable_DSF=enable_DSF
		if(self.enable_DSF):
			self.DSF=DSF(M,T)

	def forward(self, x):

		channel_scores=self.channel_scorer(x)

		selection,rate=self.gumbel(channel_scores,gumbel_noise=self.training and not self.freeze)

		rate=rate.view(-1,1,self.M,1)
		selection=selection.view(-1,1,self.M,1)

		out=x*selection

		if(self.enable_DSF):
			out=self.DSF(out)

		return out,rate


################################################################################################################################################################################3
################################################################################################################################################################################3
################################################################################################################################################################################3
################################################################################################################################################################################3
################################################################################################################################################################################3
################################################################################################################################################################################3


class GumbelSoftMax(nn.Module):
	def __init__(self, eps=1e-8):
		super(GumbelSoftMax, self).__init__()
		self.eps = eps
		self.freeze=False

	def forward(self, x, gumbel_temp=1.0, freeze=False):
	
		gumbel_hard_temp=0.1

		if(not self.training):
			return make_onehot(x).float(),make_onehot(x).float()

		rate=torch.softmax(x/gumbel_hard_temp,dim=1)

		if(freeze):
			selection=make_onehot(x).float()
		else:
			eps = self.eps
			u = torch.rand_like(x)
			g = -torch.log(-torch.log(u + eps)+eps)
			x = x + g
			selection=torch.softmax(x/gumbel_temp,dim=1)			

		return selection,rate




class StochasticGumbelSoftMax(nn.Module):
	def __init__(self, eps=1e-8):
		super(StochasticGumbelSoftMax, self).__init__()
		self.eps = eps

	def forward(self, x, gumbel_temp=1.0,freeze=False):

		gumbel_hard_temp=0.1

		eps = self.eps
		u = torch.rand_like(x)
		g = -torch.log(-torch.log(u + eps)+eps)
		x = x + g

		if(not self.training):
			return make_onehot(x).float(),make_onehot(x).float()

		if(freeze):
			selection=make_onehot(x).float()
		else:
			selection=torch.softmax(x/gumbel_temp,dim=1)	

		rate=torch.softmax(x/gumbel_hard_temp,dim=1)

		return selection,rate






class ChannelScorer_Policy(nn.Module):
	def __init__(self, M,T):

		super(ChannelScorer_Policy, self).__init__()
		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

		self.M=M
		self.FT=10
		self.FS=10


		self.conv1=nn.Conv2d(1,self.FT,kernel_size=(1,17),padding=(0,8))
		self.conv2=nn.Conv2d(self.FT,self.FS,kernel_size=(self.M,1),padding=(0,0))
		self.pooling2 = nn.AvgPool2d(kernel_size=(1, 75),stride=(1,15),padding=(0,0))

		#Temporal average pooling
		self.pooling2 = nn.AvgPool2d(kernel_size=(1, 75),stride=(1,15),padding=(0,0))

		self.fc1 = nn.Linear(int(self.FS*(1+math.floor(T-75)/15))+self.M, int(M),bias=False)
	
		self.zerostart=Parameter(torch.randn(self.M,1)*0.01)


	def forward(self, x,mask):

		if(torch.sum(torch.abs(x))<1e-16):
			#A priori scores for channels without any information present
			out=self.zerostart.view(1,-1,1).repeat(x.size(0),1,1)
		else:
			x=self.conv1(x)
			x = torch.pow(self.conv2(x),2)
			x = self.pooling2(x)
			x = torch.log(torch.clamp(x,min=epsilon))	
			# x = torch.log(x+1e-16)			
		
			# FC Layer
			x = torch.flatten(x,start_dim=1)
			x=torch.cat((x,mask.view(-1,self.M)),dim=1)
			x = self.fc1(x)
			out=x.view(-1,self.M,1)

		return out






class PolicySelector(nn.Module):
	def __init__(self, M,K,T):

		super(PolicySelector, self).__init__()

		self.M=M
		self.K=K
		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

		self.gumbel=GumbelSoftMax()
		self.stochasticgumbel=StochasticGumbelSoftMax()

		self.channel_scorer=ChannelScorer_Policy(M,T)


	def forward(self, x,mask,freeze=False,temperature=1.0):

		x_current=mask*x
		scores=self.channel_scorer(x_current,mask)
		#Make scores of already selected channels extremely low
		scores=scores-1e6*mask.view(-1,self.M,1)

		#For starting with probability distribution: always apply noise to first input
		if(torch.sum(torch.abs(x_current))<1e-16):
			selection,rate=self.stochasticgumbel(scores,freeze=freeze,gumbel_temp=temperature)
		else:
			selection,rate=self.gumbel(scores,freeze=freeze,gumbel_temp=temperature)

		return selection,rate





class CMI(nn.Module):
	
	def __init__(self,input_dim,output_dim=4,enable_DSF=False,K=2,batchnorm=False,eval_mode=False):
		super(CMI,self).__init__()
		self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

		self.M = input_dim[0]
		self.T = input_dim[1]
		self.K=K
		self.output_dim = output_dim

		self.selector=PolicySelector(self.M,self.K,self.T)
		self.predictor = MSFBCNN(input_dim=[self.M,self.T],output_dim=output_dim,batchnorm=batchnorm)

		self.enable_DSF=enable_DSF
		if(self.enable_DSF):
			self.DSF=DSF(self.M,self.T)
		self.gumbel=GumbelSoftMax()

		self.freeze=False
		self.temperature=1.0

		self.layers = self.create_layers_field()
		self.apply(init_weights)

		self.eval_mode = eval_mode
		
	def forward(self,x):

		mask=set_cuda(torch.zeros(x.size(0),1,self.M,1))

		allselections=set_cuda(torch.zeros(self.K,x.size(0),self.M,1))
		allrate=set_cuda(torch.zeros(self.K,x.size(0),self.M,1))
		allout=set_cuda(torch.zeros(self.K,x.size(0),self.output_dim))

		for k in range(self.K):

			selection,rate=self.selector(x,mask,freeze=self.freeze,temperature=self.temperature)
			# print('K = ' + str(k))
			# print('Sampled', selection[:5,:])
			# print('Rate', rate[:5,:])
			# time.sleep(10)

			#Check dimensions
			hard=make_onehot(selection)

			z=selection.view(-1,1,self.M,1)

			#Add new selection to existing mask
			z_mask=torch.max(z,mask)

			#Compute prediction for masked input
			x_sel=z_mask*x
			if(self.enable_DSF):
				x_sel=self.DSF(x_sel)

			if(self.eval_mode and not(k==self.K-1)):
				out=set_cuda(torch.randn(x.size(0),self.output_dim))
			else:
				out=self.predictor(x_sel)


			#Update obtained mask
			#Paper version: randomly sample new hard from existing logits, with current selection already subtracted -> avoids repetitions during training
			#Not sure if detach is needed, limits backpropagation to only occur within each single iteration step
		
			added_mask_hard=hard.view(-1,1,self.M,1).detach()
			mask=torch.max(mask,added_mask_hard)

			#Update all outputs
			allselections[k,:,:,:]=selection
			allrate[k,:,:,:]=rate
			allout[k,:,:]=out

		rate=torch.sum(allrate,dim=0)	

		return out,allout,rate

	def regularizer(self):
		#L2-Regularization of other layers
		reg=self.floatTensor([0])
		for i,layer in enumerate(self.layers):
				if(type(layer) == nn.Conv2d or type(layer) == nn.Linear):
					reg+=torch.sum(torch.pow(layer.weight,2))
		return reg

	def create_layers_field(self):
		layers = []
		for idx, m in enumerate(self.modules()):
			if(type(m) == nn.Conv2d or type(m) == nn.Linear or type(m) == PolicySelector):
				layers.append(m)
		return layers
	
	def set_temperature(self,temp):
		self.temperature=temp		

	def set_freeze(self,freeze):
		self.freeze=freeze
		if(freeze==True):
			self.selector.eval()
			for param in self.selector.parameters():
				param.requires_grad = False
		else:
			self.selector.train()
			for param in self.selector.parameters():
				param.requires_grad = True


def make_onehot(x):
	#Input: BxMxK
	_,ind=torch.max(x,dim=1)
	out = F.one_hot(ind,x.size(1))
	out=out.transpose(-1,-2)
	return out


