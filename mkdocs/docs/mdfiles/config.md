# config.py

This file contains the hyper- and training parameters of the ICNN for learning the hidden material model using the NN-EUCLID framework. Table A.1 in the [paper](https://doi.org/10.1016/j.jmps.2022.105076) discusses these parameters.

## Hyper- and training parameters
| Parameter | Description |
| ----------|-------------|
| `ensemble_size` |                  Number of ICNNs in the ensemble
|`random_init`|  Randomly initialize weights and biases of ICNNs using xavier_uniform
| `n_input`|                        Default = 3, i.e., the three principal invariants.
| `n_output`|                       Default = 1, i.e., the strain energy density.
|`n_hidden`|                      List of number of neurons for each hidden layer.
| `use_dropout`|      Use dropout in ICNN architecture.
| `dropout_rate`|                   Dropout probability.
| `use_sftpSquared`|                Use squared softplus activation for the hidden layers.
| `scaling_sftpSq`|               Scale the output after squared softplus activation to mitigate exploding gradients.
| `opt_method`|                   Specify the NN optimizer.
|`epochs`|                         Number of epochs to train the ICNN
| `lr_schedule`|  Choose a learning rate scheduler to improve convergence and performance.
| `eqb_loss_factor`|               Factor to scale the force residuals at the free DoFs.
| `reaction_loss_factor`|           Factor to scale the force residuals at the fixed DoFs.
| `verbose_frequency`|              Prints the training progress every $n^{th}$ ($n=$`verbose_frequency`) epoch.

## Plotting parameters
| Parameter | Description |
| ----------|-------------|
| `plot_quantities`|Which quantities to evaluate and plot.
| `strain_paths`|                   Which strain paths to evaluate and plot.
| `lw_truth`|                       Linewidth of the true strain energy density response.
| `lw_best`|                        Linewidth of the strain energy response of accepted models.
| `lw_worst`|                       Linewidth of the strain energy response of rejected models.
| `color_truth`|                    Color for the true strain energy reponse
| `color_best`|                     Color for the strain energy response of accepted models.
| `color_worst`|                    Color for the strain energy response of rejected models.
| `alpha_best`|                     Opacity of the line for the accepted models.
| `alpha_worst`|                    Opacity of the line for the rejected models.
| `g_min, g_max`|                   Range of the loading parameter gamma.
| `gamma_steps`|                    How many loading steps to take.
| `remove_ensemble_outliers`|       If categorization into accepted and rejected should be made.
| `accept_ratio`|                   Defines loss range in which accepted models fall into.
| `fs`|                             fontsize
