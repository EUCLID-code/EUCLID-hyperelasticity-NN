from core import *

def get_data_path(fem_dir, fem_material, noise_level, loadstep):

	prefix = fem_dir;
	if(prefix[-1]=='/'):
		prefix = prefix[0:-1]

	if noise_level == 'low':
		prefix = prefix + '/noise=low/'
	elif noise_level == 'high':
		prefix = prefix + '/noise=high/'
	else:
		raise ValueError('Wrong noise_level parameter provided')
	# Add fem_material
	prefix = prefix + fem_material


	fem_path = prefix+'/'+str(loadstep)+'/'

	return fem_path

def exportTensor(folder,name,data,cols, header=True):
	# print(folder)
	os.makedirs(folder,exist_ok=True)
	#export torch.tensor to pickle
	df=pd.DataFrame.from_records(data.detach().cpu().numpy())
	if(header):
		df.columns = cols
	# print(name+".csv")
	df.to_csv(folder+'/'+name+".csv",header=header,index=False)

def exportList(folder,name,data):
	# print(folder)
	os.makedirs(folder,exist_ok=True)
	#export torch.tensor to pickle
	arr=np.array(data)
	np.savetxt(folder+'/'+name+".csv", arr, delimiter=',')
