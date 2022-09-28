import sys
import matplotlib
sys.path.insert(0, '../')
from core import *
#config
from config import *
#CUDA
initCUDA(cuda)
#supporting files
from model import *
from helper import *
from matplotlib.ticker import FormatStrFormatter

matplotlib.pyplot.rcParams['font.family'] = 'serif'
matplotlib.pyplot.rcParams['mathtext.fontset'] = 'dejavuserif'


def evaluate_icnn(model, fem_material, noise_level, plot_quantities, output_dir):

	"""
	Evaluates the trained model along six deformation paths and compares it to the ground truth model.

	Arguments:

	model:				Trained model class instance.
	fem_material:		String specifying the name of the hidden material
	noise_level:		{low,high}
	plot_quantities:	Which quantites to plot {W,P}
	output_dir:			Output directory name

	"""


	model.eval()

	def compute_corrected_W(F):
		"""
		Compute the strain energy density according to Ansatz (Eq. 8).

		Input: 		Deformation gradient F in form (F11,F12,F21,F22)
		Output: 	Strain energy density according to Ansatz (Eq. 8)

		"""

		F_0 = torch.zeros((1,4))
		F_0[:,0] = 1
		F_0[:,3] = 1

		F11_0 = F_0[:,0:1]
		F12_0 = F_0[:,1:2]
		F21_0 = F_0[:,2:3]
		F22_0 = F_0[:,3:4]

		F11_0.requires_grad = True
		F12_0.requires_grad = True
		F21_0.requires_grad = True
		F22_0.requires_grad = True

		# Get components of F
		F11 = F[:,0:1]
		F12 = F[:,1:2]
		F21 = F[:,2:3]
		F22 = F[:,3:4]

		F11.requires_grad = True
		F12.requires_grad = True
		F21.requires_grad = True
		F22.requires_grad = True

		W_NN = model(torch.cat((F11,F12,F21,F22),dim=1))

		dW_NN_dF11 = torch.autograd.grad(W_NN,F11,torch.ones(F11.shape[0],1),create_graph=True)[0]
		dW_NN_dF12 = torch.autograd.grad(W_NN,F12,torch.ones(F12.shape[0],1),create_graph=True)[0]
		dW_NN_dF21 = torch.autograd.grad(W_NN,F21,torch.ones(F21.shape[0],1),create_graph=True)[0]
		dW_NN_dF22 = torch.autograd.grad(W_NN,F22,torch.ones(F22.shape[0],1),create_graph=True)[0]

		P_NN = torch.cat((dW_NN_dF11,dW_NN_dF12,dW_NN_dF21,dW_NN_dF22),dim=1)

		# Derivative of volumetric term
		J_F_inv_T = torch.cat((F22,-F21,-F12,F11),1)
		C = computeCauchyGreenStrain(torch.cat((F11,F12,F21,F22),dim=1))
		_,_,I3 = computeStrainInvariants(C)

		# Zero deformation input data
		W_NN_0 = model(torch.cat((F11_0,F12_0,F21_0,F22_0),dim=1))

		dW_NN_dF11_0 = torch.autograd.grad(W_NN_0,F11_0,torch.ones(F11_0.shape[0],1),create_graph=True)[0]
		dW_NN_dF12_0 = torch.autograd.grad(W_NN_0,F12_0,torch.ones(F12_0.shape[0],1),create_graph=True)[0]
		dW_NN_dF21_0 = torch.autograd.grad(W_NN_0,F21_0,torch.ones(F21_0.shape[0],1),create_graph=True)[0]
		dW_NN_dF22_0 = torch.autograd.grad(W_NN_0,F22_0,torch.ones(F22_0.shape[0],1),create_graph=True)[0]

		P_NN_0 = torch.cat((dW_NN_dF11_0,dW_NN_dF12_0,dW_NN_dF21_0,dW_NN_dF22_0),dim=1)

		P_cor = torch.zeros_like(P_NN)

		P_cor[:,0:1] = F11*-P_NN_0[:,0:1] + F12*-P_NN_0[:,2:3]
		P_cor[:,1:2] = F11*-P_NN_0[:,1:2] + F12*-P_NN_0[:,3:4]
		P_cor[:,2:3] = F21*-P_NN_0[:,0:1] + F22*-P_NN_0[:,2:3]
		P_cor[:,3:4] = F21*-P_NN_0[:,1:2] + F22*-P_NN_0[:,3:4]

		P = P_NN + P_cor
		E = torch.zeros_like(F)

		E[:,0:1] = 0.5*(C[:,0:1]-1.)
		E[:,1:2] = 0.5*(C[:,1:2])
		E[:,2:3] = 0.5*(C[:,2:3])
		E[:,3:4] = 0.5*(C[:,3:4]-1.)

		W_cor = torch.sum(-P_NN_0*E, 1, keepdim=True)

		# The actual offset of the NN output is W_NN_0 (NN ouput at F=I) and H:E
		W_offset = W_NN_0
		# W is sum of NN output, the volumetric correction term and subtracting NN output at F=I and H:E
		W = W_NN + W_cor - W_offset

		P = P.view(-1,4,1)

		return W, P

	def get_true_W(fem_material,J,C,I1,I2,I3):
		"""

		Analytical description of benchmark hyperelastic material models.
		Input:		Strain invariants and Cauchy-Green deformation matrix.
		Output:		Strain energy density of the specified material

		"""
		if fem_material == 'NeoHookean':
			I1_tilde = J**(-2/3)*I1
			W_truth = 0.5000*(I1_tilde - 3) + 1.5000*(J - 1)**2
		elif fem_material == 'Anisotropy45':
			I1_tilde = J**(-2/3)*I1
			alpha = torch.tensor([np.pi/4])
			C11 = C[:,0:1]
			C12 = C[:,1:2]
			C21 = C[:,2:3]
			C22 = C[:,3:4]
			Ia = torch.cos(alpha)*(C11*torch.cos(alpha)+C12*torch.sin(alpha)) + torch.sin(alpha)*(C21*torch.cos(alpha)+C22*torch.sin(alpha))
			Ia_tilde = Ia * J**(-2/3)
			W_truth = 0.5*(I1_tilde - 3.) + 0.75*(J - 1.)**2 + 0.5*(Ia_tilde - 1.)**2
		elif fem_material == 'Anisotropy60':
			I1_tilde = J**(-2/3)*I1
			alpha = torch.tensor([np.pi/3])
			C11 = C[:,0:1]
			C12 = C[:,1:2]
			C21 = C[:,2:3]
			C22 = C[:,3:4]
			Ia = torch.cos(alpha)*(C11*torch.cos(alpha)+C12*torch.sin(alpha)) + torch.sin(alpha)*(C21*torch.cos(alpha)+C22*torch.sin(alpha))
			Ia_tilde = Ia * J**(-2/3)
			W_truth = 0.5*(I1_tilde - 3.) + 0.75*(J - 1.)**2 + 0.5*(Ia_tilde - 1.)**2
		elif fem_material == 'Isihara':
			I1_tilde = J**(-2/3)*I1
			I2_tilde = J**(-4/3)*I2
			W_truth = 0.5000*(I1_tilde - 3) + 1.0000*(I2_tilde - 3) + 1.0000*(I1_tilde - 3)**2 + 1.5000*(J - 1)**2
		elif fem_material == 'HainesWilson':
			I1_tilde = J**(-2/3)*I1
			I2_tilde = J**(-4/3)*I2
			W_truth = 0.5000*(I1_tilde - 3) + 1.0000*(I2_tilde - 3) + 0.7000*(I1_tilde - 3)*(I2_tilde - 3) + 0.2000*(I1_tilde - 3)**3 + 1.5000*(J - 1)**2
		elif fem_material == 'GentThomas':
			I1_tilde = J**(-2/3)*I1
			I2_tilde = J**(-4/3)*I2
			W_truth = 0.5000*(I1_tilde - 3) + 1.5000*(J - 1)**2 + 1.0000*torch.log(I2_tilde/3)
		elif fem_material == 'ArrudaBoyce':
			shear_modulus = 2.5
			chain_length = 28
			kappa = 1.5
			shearModulus = torch.tensor([shear_modulus])
			N = torch.tensor([chain_length])
			sqrt_N = torch.sqrt(N)
			W_truth = torch.zeros((gamma_steps,1))
			I1_tilde = J**(-2/3)*I1
			lambda_chain = 1
			x = lambda_chain / torch.sqrt(N)
			beta_chain = 0.0
			if torch.abs(x)<0.841:
				beta_chain = 1.31*torch.tan(1.59*x)+0.91*x
			else:
				beta_chain = 1./((x+0.00000001/(torch.abs(x)+0.00000001))-x)
			W_truth_offset = shearModulus * sqrt_N * (beta_chain*lambda_chain+sqrt_N*torch.log(beta_chain/torch.sinh(beta_chain)))
			for step in range(gamma_steps):
				lambda_chain = torch.sqrt(I1_tilde[step:step+1]/3.)
				x = lambda_chain / torch.sqrt(N)
				beta_chain = 0.0
				if torch.abs(x)<0.841:
					beta_chain = 1.31*torch.tan(1.59*x)+0.91*x
				else:
					beta_chain = 1./((x+0.00000001/(torch.abs(x)+0.00000001))-x)
				#% Final output
				W_truth[step:step+1,:] = shearModulus * sqrt_N * (beta_chain*lambda_chain+sqrt_N*torch.log(beta_chain/torch.sinh(beta_chain))) - W_truth_offset + kappa*(J[step:step+1]-1)**2

		elif fem_material == 'Ogden':
			kappa_ogden = 1.5
			mu_ogden = 0.65
			alpha_ogden = 0.65
			I1_tilde = J**(-2/3)*I1 + 1e-13
			I1t_0 =torch.tensor([3]) + 1e-13
			J_0 = torch.tensor([1]) + 1e-13
			W_offset = kappa_ogden*(J_0-1)**2 + 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1t_0  +  torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.)) - 1/(J_0**(2./3.)) )**alpha_ogden+( 0.5*I1t_0 - 0.5*torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.))  - 0.5/(J_0**(2./3.)) )**alpha_ogden + J_0**(-alpha_ogden*2./3.) ) * mu_ogden
			#% Final output
			W_truth = kappa_ogden*(J-1)**2 + 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1_tilde  +  torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.)) - 1/(J**(2./3.)) )**alpha_ogden+( 0.5*I1_tilde - 0.5*torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.))  - 0.5/(J**(2./3.)) )**alpha_ogden + J**(-alpha_ogden*2./3.) ) * mu_ogden - W_offset

		elif fem_material == 'Holzapfel':
			I1_tilde = J**(-2/3)*I1
			alpha = torch.tensor([np.pi/6])
			beta = -1.*alpha
			C11 = C[:,0:1]
			C12 = C[:,1:2]
			C21 = C[:,2:3]
			C22 = C[:,3:4]
			Ia = torch.cos(alpha)*(C11*torch.cos(alpha)+C12*torch.sin(alpha)) + torch.sin(alpha)*(C21*torch.cos(alpha)+C22*torch.sin(alpha))
			Ib = torch.cos(beta)*(C11*torch.cos(beta)+C12*torch.sin(beta)) + torch.sin(beta)*(C21*torch.cos(beta)+C22*torch.sin(beta))
			Ia_tilde = Ia * J**(-2/3)
			Ib_tilde = Ib * J**(-2/3)
			k1h = 0.9
			k2h = 0.8
			HAa = k1h/2./k2h*(torch.exp(k2h*torch.pow(Ia_tilde - 1.,2.))-1.)
			HAb = k1h/2./k2h*(torch.exp(k2h*torch.pow(Ib_tilde - 1.,2.))-1.)
			#% Final output
			W_truth = 0.5*(I1_tilde-3) + 1.0*(J-1)**2 + HAa + HAb

		return W_truth

	for plot_quantity in plot_quantities:

		fig, axs = matplotlib.pyplot.subplots(3,3)
		fig.set_figwidth(9)
		fig.set_figheight(9)

		for i in range(3):
			fig.delaxes(axs[i,-1])

		axs_i = 0
		axs_j = 0

		for path_count, strain_path in enumerate(strain_paths):
			if strain_path == 'UT':
				P_idx = 0
			elif strain_path == 'SS':
				P_idx = 1
			elif strain_path == 'PS':
				P_idx = 3
			elif strain_path == 'UC':
				P_idx = 0
			elif strain_path == 'BT':
				P_idx = 0
			elif strain_path == 'BC':
				P_idx = 0

			print('Evaluating and plotting: '+fem_material+' for strain path '+strain_path)

			gamma=np.linspace(g_min,g_max,gamma_steps)
			F, xlabel = getStrainPathDeformationGradient(strain_path, gamma_steps, gamma)

			# Get components of F
			F11 = F[:,0:1]
			F12 = F[:,1:2]
			F21 = F[:,2:3]
			F22 = F[:,3:4]

			F11.requires_grad = True
			F12.requires_grad = True
			F21.requires_grad = True
			F22.requires_grad = True

			#computing detF
			J = computeJacobian(torch.cat((F11,F12,F21,F22),dim=1))

			#computing Cauchy-Green strain: C = F^T F
			C = computeCauchyGreenStrain(torch.cat((F11,F12,F21,F22),dim=1))

			#computing strain invariants
			I1, I2, I3 = computeStrainInvariants(C)

			#Get true model of fem_material
			W_truth = get_true_W(fem_material,J,C,I1,I2,I3)

			dW_truth_dF11 = torch.autograd.grad(W_truth,F11,torch.ones(F11.shape[0],1),create_graph=True)[0]
			dW_truth_dF12 = torch.autograd.grad(W_truth,F12,torch.ones(F12.shape[0],1),create_graph=True)[0]
			dW_truth_dF21 = torch.autograd.grad(W_truth,F21,torch.ones(F21.shape[0],1),create_graph=True)[0]
			dW_truth_dF22 = torch.autograd.grad(W_truth,F22,torch.ones(F22.shape[0],1),create_graph=True)[0]

			P_truth = torch.cat((dW_truth_dF11,dW_truth_dF12,dW_truth_dF21,dW_truth_dF22),dim=1)

			# Define plot bounds
			y_max_P = (torch.max(P_truth[:,P_idx])*1.1).detach().numpy()
			if y_max_P <= 0.:
				y_min_P = (torch.min(P_truth[:,P_idx])*1.1).detach().numpy()
				y_max_P = 0.0
			else:
				y_min_P = 0.

			if remove_ensemble_outliers:
				final_losses = torch.zeros((ensemble_size,1))
				for ensemble_iter in range(ensemble_size):
					final_losses[ensemble_iter] = pd.read_csv(output_dir+'/'+fem_material+'/loss_history_noise='+noise_level+'_ID='+str(ensemble_iter)+'.csv', header=None).values[-1][1]

				final_losses_ratio = final_losses / torch.min(final_losses)
				num_models_remove = torch.where(final_losses_ratio >= accept_ratio)[0].shape[0]
				num_models_keep = ensemble_size - num_models_remove

				idx_best_models = torch.topk(-final_losses.flatten(),num_models_keep).indices
				idx_worst_models = torch.topk(final_losses.flatten(),num_models_remove).indices

			W_predictions = torch.zeros((gamma_steps,ensemble_size))
			P_predictions = torch.zeros((gamma_steps,4,ensemble_size))

			alpha_values = np.zeros(ensemble_size)
			for ensemble_iter in range(ensemble_size):
				model.load_state_dict(torch.load(output_dir+'/'+fem_material+'/noise='+noise_level+'_ID='+str(ensemble_iter)+'.pth'))
				if model.anisotropy_flag is not None:
					if model.anisotropy_flag == 'single':
						alpha_values[ensemble_iter] = torch.sigmoid(model.alpha).detach()*180
					elif model.anisotropy_flag == 'double':
						alpha_values[ensemble_iter] = torch.sigmoid(model.alpha).detach()*90

				W_predictions[:,ensemble_iter:ensemble_iter+1], P_predictions[:,:,ensemble_iter:ensemble_iter+1] = compute_corrected_W(F)

			P_accepted = P_predictions[:,:,idx_best_models]
			P_rejected = P_predictions[:,:,idx_worst_models]

			W_accepted = W_predictions[:,idx_best_models]
			W_rejected = W_predictions[:,idx_worst_models]


			if path_count < 3:
				axs_i = path_count
				axs_j = 0
			else:
				axs_i = path_count-3
				axs_j = 1

			if plot_quantity == 'P':
				if path_count == 0:
					accepted, = axs[axs_i,axs_j].plot(gamma,P_accepted[:,P_idx,0].detach().numpy(), color=color_accepted, linestyle='--', lw=lw_best,alpha=alpha_best)
					if len(idx_best_models)>2:
						axs[axs_i,axs_j].plot(gamma,P_accepted[:,P_idx,-(len(idx_best_models)-1)::].detach().numpy(), color=color_accepted, linestyle='--', lw=lw_best,alpha=alpha_best)
					else:
						axs[axs_i,axs_j].plot(gamma,P_accepted[:,P_idx,:1].detach().numpy(), color=color_accepted, linestyle='--', lw=lw_best,alpha=alpha_best)
				else:
					axs[axs_i,axs_j].plot(gamma,P_accepted[:,P_idx].detach().numpy(), color=color_accepted, linestyle='--', lw=lw_best,alpha=alpha_best)

				if P_rejected.shape[2] > 0:
					if path_count == 0 :
						rejected, = axs[axs_i,axs_j].plot(gamma,P_rejected[:,P_idx,0].detach().numpy(), color=color_rejected, linestyle='dotted', lw=lw_worst,alpha=alpha_worst)
						if len(idx_worst_models)>2:
							axs[axs_i,axs_j].plot(gamma,P_rejected[:,P_idx,-(len(idx_worst_models)-1)::].detach().numpy(), color=color_rejected, linestyle='dotted', lw=lw_worst,alpha=alpha_worst)
						else:
							axs[axs_i,axs_j].plot(gamma,P_rejected[:,P_idx,:1].detach().numpy(), color=color_rejected, linestyle='dotted', lw=lw_worst,alpha=alpha_worst)
					else:
						axs[axs_i,axs_j].plot(gamma,P_rejected[:,P_idx].detach().numpy(), color=color_rejected, linestyle='dotted', lw=lw_worst,alpha=alpha_worst)

				if path_count == 0:
					truth, = axs[axs_i,axs_j].plot(gamma,P_truth[:,P_idx].detach().numpy(),color=color_truth,linestyle='-', lw=lw_truth)
				else:
					axs[axs_i,axs_j].plot(gamma,P_truth[:,P_idx].detach().numpy(),color=color_truth,linestyle='-', lw=lw_truth)

				if P_idx == 0:
					axs[axs_i,axs_j].set_ylabel(strain_path+'\n'+r'$P_{11}(\gamma)$',fontsize=fs)
				elif P_idx == 1:
					axs[axs_i,axs_j].set_ylabel(strain_path+'\n'+r'$P_{12}(\gamma)$',fontsize=fs)
				elif P_idx == 3:
					axs[axs_i,axs_j].set_ylabel(strain_path+'\n'+r'$P_{22}(\gamma)$',fontsize=fs)

				axs[axs_i,axs_j].set_ylim([y_min_P,y_max_P])
				axs[axs_i,axs_j].set_aspect(g_max/(y_max_P-y_min_P))
				axs[axs_i,axs_j].set_yticks([y_min_P,(y_max_P+y_min_P)/2,y_max_P])

				axs[axs_i,axs_j].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
				axs[axs_i,axs_j].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
				axs[axs_i,axs_j].set_xlim([g_min,g_max])
				if path_count == len(strain_paths)-1:
					axs[axs_i,axs_j].set_xlabel(r'$\gamma$',fontsize=fs)

				axs[axs_i,axs_j].set_xticks([0,g_max/2,g_max])
				axs[axs_i,axs_j].tick_params(axis='both', labelsize=fs)

			elif plot_quantity == 'W':

				y_min_W = (torch.min(W_truth)).detach().numpy()
				y_max_W = (torch.max(W_truth)*1.1).detach().numpy()

				if path_count == 0:
					accepted, = axs[axs_i,axs_j].plot(gamma,W_accepted[:,0].detach().numpy(), color=color_accepted, linestyle='--', lw=lw_best,alpha=alpha_best)
					if len(idx_best_models)>2:
						axs[axs_i,axs_j].plot(gamma,W_accepted[:,-(len(idx_best_models)-1)::].detach().numpy(), color=color_accepted, linestyle='--', lw=lw_best,alpha=alpha_best)
					else:
						axs[axs_i,axs_j].plot(gamma,W_accepted[:,:1].detach().numpy(), color=color_accepted, linestyle='--', lw=lw_best,alpha=alpha_best)
				else:
					axs[axs_i,axs_j].plot(gamma,W_accepted.detach().numpy(), color=color_accepted, linestyle='--', lw=lw_best,alpha=alpha_best)

				if W_rejected.shape[1] > 0:
					if path_count == 0:
						rejected, = axs[axs_i,axs_j].plot(gamma,W_rejected[:,0].detach().numpy(), color=color_rejected, linestyle='dotted', lw=lw_worst,alpha=alpha_worst)
						if len(idx_worst_models)>2:
							axs[axs_i,axs_j].plot(gamma,W_rejected[:,-(len(idx_worst_models)-1)::].detach().numpy(), color=color_rejected, linestyle='dotted', lw=lw_worst,alpha=alpha_worst)
						else:
							axs[axs_i,axs_j].plot(gamma,W_rejected[:,:1].detach().numpy(), color=color_rejected, linestyle='dotted', lw=lw_worst,alpha=alpha_worst)
					else:
						axs[axs_i,axs_j].plot(gamma,W_rejected.detach().numpy(), color=color_rejected, linestyle='dotted', lw=lw_worst,alpha=alpha_worst)

				if path_count == 0:
					truth, = axs[axs_i,axs_j].plot(gamma,W_truth.detach().numpy(),color=color_truth,linestyle='-', lw=lw_truth)
				else:
					axs[axs_i,axs_j].plot(gamma,W_truth.detach().numpy(),color=color_truth,linestyle='-', lw=lw_truth)

				axs[axs_i,axs_j].set_ylabel(strain_path+'\n'+r'$W(\mathbf{F}(\gamma))$',fontsize=fs)

				axs[axs_i,axs_j].set_ylim([y_min_W,y_max_W])
				axs[axs_i,axs_j].set_aspect(g_max/(y_max_W-y_min_W))
				axs[axs_i,axs_j].set_yticks([y_min_W,(y_max_W+y_min_W)/2,y_max_W])

				axs[axs_i,axs_j].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
				axs[axs_i,axs_j].xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
				axs[axs_i,axs_j].set_xlim([g_min,g_max])
				if path_count == len(strain_paths)-1:
					axs[axs_i,axs_j].set_xlabel(r'$\gamma$',fontsize=fs)

				axs[axs_i,axs_j].set_xticks([0,g_max/2,g_max])
				axs[axs_i,axs_j].tick_params(axis='both', labelsize=fs)

			if W_rejected.shape[1] > 0:
				lgd = fig.legend(handles=[truth,accepted,rejected],labels=['True','Accepted','Rejected'],loc='upper right', bbox_to_anchor=(0.9, 0.9),fontsize=fs)
			else:
				lgd = fig.legend(handles=[truth,accepted],labels=['True','Accepted'],loc='upper right', bbox_to_anchor=(0.9, 0.9),fontsize=fs)

			if model.anisotropy_flag is not None:

				x = np.linspace(-1,1,100)
				if fem_material == 'Anisotropy45':
					alpha_GT = 45*np.pi/180
				elif fem_material == 'Anisotropy60':
					alpha_GT = 60*np.pi/180
				elif fem_material == 'Holzapfel':
					alpha_GT = 30*np.pi/180

				alpha_accepted = alpha_values[idx_best_models]
				alpha_rejected = alpha_values[idx_worst_models]

				theta_GT = alpha_GT*np.ones_like(x)
				theta_HA = -alpha_GT*np.ones_like(x)
				ax_polar = fig.add_subplot(3,3,6, projection='polar')
				ax_polar.plot(theta_GT,x,label='True',lw=lw_truth,color=color_truth)
				ax_polar.plot(theta_GT+np.pi,x,label='True',lw=lw_truth,color=color_truth)

				if len(alpha_accepted.shape) == 0:

					if fem_material == 'Holzapfel':
						ax_polar.plot(-theta_GT,x,label='True',lw=lw_truth,color=color_truth)
						ax_polar.plot(-theta_GT+np.pi,x,label='True',lw=lw_truth,color=color_truth)

					theta_accepted = alpha_accepted*np.pi/180*np.ones_like(x)
					ax_polar.plot(theta_accepted,x,label='Accepted',lw=lw_best,color=color_accepted,alpha=alpha_best,ls='--')
					ax_polar.plot(theta_accepted+np.pi,x,lw=lw_best,color=color_accepted,alpha=alpha_best,ls='--')
					if fem_material == 'Holzapfel':
						ax_polar.plot(-theta_accepted,x,lw=lw_best,color=color_accepted,alpha=alpha_best,ls='--')
						ax_polar.plot(-theta_accepted+np.pi,x,lw=lw_best,color=color_accepted,alpha=alpha_best,ls='--')

				else:
					if fem_material == 'Holzapfel':
						ax_polar.plot(-theta_GT,x,label='True',lw=lw_truth,color=color_truth)
						ax_polar.plot(-theta_GT+np.pi,x,label='True',lw=lw_truth,color=color_truth)

					for count,alpha_a in enumerate(alpha_accepted):
						theta_accepted = alpha_a*np.pi/180*np.ones_like(x)
					if count == 0:
						ax_polar.plot(theta_accepted,x,label='Accepted',lw=lw_best,color=color_accepted,alpha=alpha_best,ls='--')
						ax_polar.plot(theta_accepted+np.pi,x,lw=lw_best,color=color_accepted,alpha=alpha_best,ls='--')
					else:
						ax_polar.plot(theta_accepted,x,lw=lw_best,color=color_accepted,alpha=alpha_best,ls='--')
						ax_polar.plot(theta_accepted+np.pi,x,lw=lw_best,color=color_accepted,alpha=alpha_best,ls='--')
					if fem_material == 'Holzapfel':
						ax_polar.plot(-theta_accepted,x,lw=lw_best,color=color_accepted,alpha=alpha_best,ls='--')
						ax_polar.plot(-theta_accepted+np.pi,x,lw=lw_best,color=color_accepted,alpha=alpha_best,ls='--')

				if len(alpha_rejected.shape) > 0:
					for count,alpha_r in enumerate(alpha_rejected):
						theta_rejected = alpha_r*np.pi/180*np.ones_like(x)
					if count == 0:
						ax_polar.plot(theta_rejected,x,label='Rejected',lw=lw_worst,color=color_rejected,alpha=alpha_worst,ls='dotted')
						ax_polar.plot(theta_rejected+np.pi,x,lw=lw_worst,color=color_rejected,alpha=alpha_worst,ls='dotted')
					else:
						ax_polar.plot(theta_rejected,x,lw=lw_worst,color=color_rejected,alpha=alpha_worst,ls='dotted')
						ax_polar.plot(theta_rejected+np.pi,x,lw=lw_worst,color=color_rejected,alpha=alpha_worst,ls='dotted')
					if fem_material == 'Holzapfel':
						ax_polar.plot(-theta_rejected,x,lw=lw_worst,color=color_rejected,alpha=alpha_worst,ls='dotted')
						ax_polar.plot(-theta_rejected+np.pi,x,lw=lw_worst,color=color_rejected,alpha=alpha_worst,ls='dotted')

				ax_polar.set_rmax(1)
				ax_polar.set_rmin(0)
				ax_polar.set_rticks([])
				ax_polar.grid(True)

			suptitle = fig.suptitle(fem_material,fontsize=15)
			fig.tight_layout()

			if plot_quantity == 'P':
				matplotlib.pyplot.savefig(output_dir+'/models_'+fem_material+'_noise='+noise_level+'_Pij_.pdf',transparent=True,bbox_extra_artists=(lgd,suptitle,), bbox_inches='tight')
			elif plot_quantity == 'W':
				matplotlib.pyplot.savefig(output_dir+'/models_'+fem_material+'_noise='+noise_level+'_W.pdf',transparent=True,bbox_extra_artists=(lgd,suptitle,), bbox_inches='tight')
