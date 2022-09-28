# utilities.py

#### `computeCauchyGreenStrain(F):`

Compute right Cauchy-Green strain tensor from deformation gradient.

_Input Arguments_

- `F` - deformation gradient in Voigt notation

_Output Arguments_

- `C` - Cauchy-Green strain tensor in Voigt notation

---


#### `computeJacobian(F):`

Compute Jacobian from deformation gradient.

_Input Arguments_

- `F` - deformation gradient in Voigt notation

_Output Arguments_

- `J` - Jacobian

---


#### `computeStrainInvariantDerivatives(F,i,secondDerivative=False):`

Compute derivatives of the invariants of the Cauchy-Green strain tensor with respect to the deformation gradient.
Plane strain is assumed.

_Input Arguments_

- `F` - deformation gradient in Voigt notation

- `i` - specify the invariant that should be differentiated- I1, I2, I3, Ia or Ib

- `secondDerivative` - specify if second derivative should be computed

_Output Arguments_

- `dIdF` - derivative (note that the size of `dIdF` depends on the choice of `secondDerivative`)

---


#### `computeStrainInvariants(C):`

Compute invariants of the Cauchy-Green strain tensor.
Plane strain is assumed.

_Input Arguments_

- `C` - Cauchy-Green strain tensor in Voigt notation

_Output Arguments_

- `I1` - 1st invariant

- `I2` - 2nd invariant

- `I3` - 3rd invariant

---

#### `getStrainPathDeformationGradient(strain_path, gamma_steps, gamma):`


Generate deformation gradients based on a given strain path.

_Input Arguments_

- `strain path` - A two letter string defining one of the elementary strain strain_paths
- `gamma`		- Gamma loading parameter for the deformation gradient.
- `gamma steps` - Specifies how many steps should be taken from F(gamma=0) to F(gamma=gamma)

_Output Arguments_

- `F` 		- Deformation gradient in Voigt notation.

- `x-label` - Label for plotting
