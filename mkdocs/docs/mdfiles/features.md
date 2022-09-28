# features.py

#### `computeFeatures_torch(I1, I2, I3):`

Transforms the three principal invariants of the Cauchy-Green deformation tensor into mixed-deviatoric-volumetric invariants and returns them in a concatenated array.

- $J = \det(\mathbf{F}) = I_3^{1/2}$
- $\tilde {I_1} = J^{-2/3}I_1$
- $\tilde {I_2} = J^{-4/3}I_2$

_Input Arguments_

- `I1` - 1st invariant

- `I2` - 2nd invariant

- `I3` - 3rd invariant

_Output Arguments_

- `x` - $[\tilde {I_1}, \tilde {I_2}, J]$

#### `getNumberOfFeatures():`

Compute number of features.

_Input Arguments_

- _none_

_Output Arguments_

- `features.shape[1]` - number of features

---
