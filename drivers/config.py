# Set to 0 to use GPU-accelerated training
cuda = -1

dim = 2
num_nodes_per_element = 3
voigt_map = [[0,1],[2,3]]

fem_dir = '../fem-data/'
output_dir = 'results'

# Dataset settings
normalization_flag = False
additional_noise = 0.


"""
-------------Neural network training settings-------------

ensemble_size:                  Number of ICNNs in the ensemble
random_init: {True,False}       Randomly initialize weights and biases of ICNNs using xavier_uniform
n_input:                        Default = 3, i.e., the three principal invariants.
n_output:                       Default = 1, i.e., the strain energy density.
n_hidden:                       List of number of neurons for each hidden layer.
use_dropout: {True,False}       Use dropout in ICNN architecture.
dropout_rate:                   Dropout probability.
use_sftpSquared:                Use squared softplus activation for the hidden layers.
scaling_sftpSq:                 Scale the output after squared softplus activation to mitigate exploding gradients.
opt_method:                     Specify the NN optimizer.
epochs:                         Number of epochs to train the ICNN
lr_schedule: {cyclic,multistep} Choose a learning rate scheduler to improve convergence and performance.
eqb_loss_factor:                Factor to scale the force residuals at the free DoFs.
reaction_loss_factor:           Factor to scale the force residuals at the fixed DoFs.
verbose_frequency:              Prints the training progress every nth epoch.
"""
ensemble_size = 30
random_init = True
n_input = 3
n_output = 1
n_hidden = [64,64,64]
use_dropout = True
dropout_rate = 0.2
use_sftpSquared = True
scaling_sftpSq = 1./12
opt_method = 'adam'
epochs = 500
lr_schedule = 'cyclic'
if lr_schedule == 'multistep':
    lr = 0.1
    lr_milestones = [epochs//4,epochs//4*2,epochs//4*3]
    lr_decay = 0.1
    cycle_momentum = False
elif lr_schedule == 'cyclic':
    base_lr = 0.001
    lr = base_lr
    max_lr = 0.1
    cycle_momentum = False
    step_size_up=50
    step_size_down=50
eqb_loss_factor = 1.
reaction_loss_factor = 1.
verbose_frequency = 1

"""
-------------Plot settings-------------

plot_quantities: {W,P}          Which quantities to evaluate and plot.
strain_paths:                   Which strain paths to evaluate and plot.
lw_truth:                       Linewidth of the true strain energy density response.
lw_best:                        Linewidth of the strain energy response of accepted models.
lw_worst:                       Linewidth of the strain energy response of rejected models.
color_truth:                    Color for the true strain energy reponse
color_best:                     Color for the strain energy response of accepted models.
color_worst:                    Color for the strain energy response of rejected models.
alpha_best:                     Opacity of the line for the accepted models.
alpha_worst:                    Opacity of the line for the rejected models.
g_min, g_max:                   Range of the loading parameter gamma.
gamma_steps:                    How many loading steps to take.
remove_ensemble_outliers:       If categorization into accepted and rejected should be made.
accept_ratio:                   Defines loss range in which accepted models fall into.
fs:                             fontsize
"""
plot_quantities = ['W','P']
strain_paths = ['UT','UC','PS','BC','BT','SS']
lw_truth = 2.
lw_best = 1.
lw_worst = 1.
color_truth = 'darkviolet'
color_accepted = 'green'
color_rejected = 'red'
alpha_best = 0.6
alpha_worst = 0.2
g_min = 0.0
g_max = 1.0
gamma_steps = 200
remove_ensemble_outliers = True
if remove_ensemble_outliers:
    accept_ratio = 1.040
fs = 10
