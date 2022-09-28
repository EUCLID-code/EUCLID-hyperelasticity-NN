from core import *
from config import *


def train_weak(model, datasets, fem_material, noise_level):

	#loss history
	loss_history = []

	# optimizer
	if(opt_method == 'adam'):
		optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	elif(opt_method == 'lbfgs'):
		optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, line_search_fn='strong_wolfe')
	elif(opt_method == 'sgd'):
		optimizer = torch.optim.SGD(model.parameters(), lr=lr)
	elif(opt_method == 'rmsprop'):
		optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
	else:
		raise ValueError('Incorrect choice of optimizer')

	#scheduler
	if lr_schedule == 'multistep':
		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,lr_milestones,lr_decay,last_epoch=-1)
	elif lr_schedule == 'cyclic':
		scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, cycle_momentum=cycle_momentum, step_size_up=step_size_up, step_size_down=step_size_down)
	else:
		print('Incorrect scheduler. Choose between `multistep` and `cyclic`.')


	if model.anisotropy_flag is not None:
		print('--------------------------------------------------------------------------------------------------------')
		print('| epoch x/xxx |   lr    |    loss    |     eqb    |  reaction  |    angle   |')
		print('--------------------------------------------------------------------------------------------------------')
	else:
		print('--------------------------------------------------------------------------------------------------------')
		print('| epoch x/xxx |   lr    |    loss    |     eqb    |  reaction  |')
		print('--------------------------------------------------------------------------------------------------------')


	for epoch_iter in range(epochs):

		def closure():

			optimizer.zero_grad()

			loss = torch.tensor([0.])

			def computeLosses(data, model):

				# Zero deformation gradient dummy data
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

				# Get components of F from dataset
				F11 = data.F[:,0:1]
				F12 = data.F[:,1:2]
				F21 = data.F[:,2:3]
				F22 = data.F[:,3:4]

				# Allow to build computational graph
				F11.requires_grad = True
				F12.requires_grad = True
				F21.requires_grad = True
				F22.requires_grad = True

				# Forward pass NN to obtain W_NN
				W_NN = model(torch.cat((F11,F12,F21,F22),dim=1))

				# Get gradients of W w.r.t F
				dW_NN_dF11 = torch.autograd.grad(W_NN,F11,torch.ones(F11.shape[0],1),create_graph=True)[0]
				dW_NN_dF12 = torch.autograd.grad(W_NN,F12,torch.ones(F12.shape[0],1),create_graph=True)[0]
				dW_NN_dF21 = torch.autograd.grad(W_NN,F21,torch.ones(F21.shape[0],1),create_graph=True)[0]
				dW_NN_dF22 = torch.autograd.grad(W_NN,F22,torch.ones(F22.shape[0],1),create_graph=True)[0]

				# Assemble First Piola-Kirchhoff stress components
				P_NN = torch.cat((dW_NN_dF11,dW_NN_dF12,dW_NN_dF21,dW_NN_dF22),dim=1)

				# Forward pass to obtain zero deformation energy correction
				W_NN_0 = model(torch.cat((F11_0,F12_0,F21_0,F22_0),dim=1))

				# Get gradients of W_NN_0 w.r.t F
				dW_NN_dF11_0 = torch.autograd.grad(W_NN_0,F11_0,torch.ones(F11_0.shape[0],1),create_graph=True)[0]
				dW_NN_dF12_0 = torch.autograd.grad(W_NN_0,F12_0,torch.ones(F12_0.shape[0],1),create_graph=True)[0]
				dW_NN_dF21_0 = torch.autograd.grad(W_NN_0,F21_0,torch.ones(F21_0.shape[0],1),create_graph=True)[0]
				dW_NN_dF22_0 = torch.autograd.grad(W_NN_0,F22_0,torch.ones(F22_0.shape[0],1),create_graph=True)[0]

				# Get stress at zero deformation
				P_NN_0 = torch.cat((dW_NN_dF11_0,dW_NN_dF12_0,dW_NN_dF21_0,dW_NN_dF22_0),dim=1)

				# Initialize zero stress correction term
				P_cor = torch.zeros_like(P_NN)

				# Compute stress correction components according to Ansatz
				P_cor[:,0:1] = F11*-P_NN_0[:,0:1] + F12*-P_NN_0[:,2:3]
				P_cor[:,1:2] = F11*-P_NN_0[:,1:2] + F12*-P_NN_0[:,3:4]
				P_cor[:,2:3] = F21*-P_NN_0[:,0:1] + F22*-P_NN_0[:,2:3]
				P_cor[:,3:4] = F21*-P_NN_0[:,1:2] + F22*-P_NN_0[:,3:4]

				# Compute final stress (NN + correction)
				P = P_NN + P_cor

				# compute internal forces on nodes
				f_int_nodes = torch.zeros(data.numNodes,dim)
				for a in range(num_nodes_per_element):
					for i in range(dim):
						for j in range(dim):
							force = P[:,voigt_map[i][j]] * data.gradNa[a][:,j] * data.qpWeights
							f_int_nodes[:,i].index_add_(0,data.connectivity[a],force)

				# clone f_int_nodes
				f_int_nodes_clone = f_int_nodes.clone()
				# set force on Dirichlet BC nodes to zero
				f_int_nodes_clone[data.dirichlet_nodes] = 0.
				# loss for force equillibrium
				eqb_loss = torch.sum(f_int_nodes_clone**2)

				reaction_loss = torch.tensor([0.])
				for reaction in data.reactions:
					reaction_loss += (torch.sum(f_int_nodes[reaction.dofs]) - reaction.force)**2

				return eqb_loss, reaction_loss

			# Compute loss for each displacement snapshot in dataset and add them together
			for data in datasets:
				eqb_loss, reaction_loss = computeLosses(data, model)
				loss += eqb_loss_factor * eqb_loss + reaction_loss_factor * reaction_loss

			# back propagate
			loss.backward()

			return loss, eqb_loss, reaction_loss

		loss, eqb_loss, reaction_loss = optimizer.step(closure)
		scheduler.step()


		if(epoch_iter % verbose_frequency == 0):
			if model.anisotropy_flag is not None:
				if model.anisotropy_flag == 'double':
					print('| epoch %d/%d | %.1E | %.4E | %.4E | %.4E | %5.6f' % (
						epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item(),torch.sigmoid(model.alpha)*90))
				elif model.anisotropy_flag == 'single':
					print('| epoch %d/%d | %.1E | %.4E | %.4E | %.4E | %5.6f' % (
						epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item(),torch.sigmoid(model.alpha)*180))
			else:
				print('| epoch %d/%d | %.1E | %.4E | %.4E | %.4E' % (
					epoch_iter+1, epochs, optimizer.param_groups[0]['lr'], loss.item(), eqb_loss.item(), reaction_loss.item()))

			loss_history.append([epoch_iter+1,loss.item()])

	return model, loss_history
