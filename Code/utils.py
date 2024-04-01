import json
import os
import torch

def json_write(filename,output_dict):

    if not os.path.isfile(filename):
        with open(filename, "w") as outfile:
            json.dump([output_dict], outfile,indent=2)
    else:
        with open(filename, 'r') as outfile:
            full_list = json.load(outfile)

        full_list.append(output_dict)
        with open(filename, mode='w') as outfile:
            json.dump(full_list, outfile,indent=2)


#Create a vector of length epochs, decaying start_value to end_value exponentially, reaching end_value at end_epoch
def exponential_decay_schedule(start_value,end_value,epochs,end_epoch):
	t = torch.FloatTensor(torch.arange(0.0,epochs))
	p = torch.clamp(t/end_epoch,0,1)
	out = start_value*torch.pow(end_value/start_value,p)

	return out


def apply_noise(data,noise_prob):

	noise_var=torch.FloatTensor(1,1,data.size(2),1).uniform_(0, 3)
	corrupted_data=noise_var*torch.randn(data.size())

	enable_noise=torch.ones(data.size(0),1,data.size(2),1)*noise_prob
	mask=torch.bernoulli(enable_noise)

	out_data=torch.where(mask==0,data,corrupted_data)

	return out_data