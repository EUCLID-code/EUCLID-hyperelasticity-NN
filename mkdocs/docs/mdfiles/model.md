# model.py

Here we define custom PyTorch classes used in our NN-EUCLID framework.

#### `class convexLinear(torch.nn.Module):`

Custom linear layer with enforced positive weights and no bias.
The operation is done as follows:

- $z = softplus(W)*x$

where $W$ contains `size_in*size_out` trainable parameters.

_Initialization arguments:_

- `size_in` -  Input dimension

- `size_out`- Output dimension

_Input arguments:_

- `x` - input data

_Output arguments:_

- `z` - linear transformation of x

---

#### `class ICNN(torch.nn.Module):`

Material model based on Input convex neural network.

_Initialization arguments:_

- `n_input` -		Input layer size
- `n_hidden` - 			List with number of neurons for each layer
- `n_output` -			Output layer size
- `use_dropout` -		Activate dropout during training
- `dropout_rate` - 	Dropout probability.
- `anisotropy_flag` -	Possible arguments: {`single`, `double`} -> type of fiber families
- `fiber_type` -			Possible arguments: {`mirror`, `general`} -> type of fiber arrangement in case of two (or more) fiber families. In case of `mirror` the second fiber is set as: $\alpha_2 = -\alpha_1$. In case of `general` the second fiber is set as: $\alpha_2 = \alpha_1+90°$.

_Input arguments:_ 			

- `x` - Deformation gradient in the form: `(F11,F12,F21,F22)`


_Output arguments:_ 			

- `W_NN` = NN-based strain energy density
