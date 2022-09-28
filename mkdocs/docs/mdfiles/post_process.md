# post_process.py

This file contains scripts to evaluate trained ICNN models and visualize their performances.

#### `evaluate_icnn(model, fem_material, noise_level, plot_quantities, output_dir)`:

Evaluates the trained model along six deformation paths and compares it to the ground truth model.

_Input arguments_:

- `model`				- Trained model class instance.
- `fem_material`	- 	String specifying the name of the hidden material
- `noise_level` - Possible arguments:{`low`,`high`}
- `plot_quantities` -	Possible arguments: {`W`,`P`}. Defined in `config.py`.
- `output_dir` - Output directory name (defined in `config.py`)

_Output arguments:_

- Plot(s) will be saved evaluating the performance of ICNN-based model against the ground truth model of `fem_material` along six deformation paths.

---

#### `compute_corrected_W(F)`:

Computes the strain energy density according to Ansatz (Eq. 8) using the trained model instance inside `evaluate_icnn()` function call.

_Input arguments_:

- `F` - Deformation gradient F in form `(F11,F12,F21,F22)`


_Output arguments_:

-	`W` - Strain energy density according to Ansatz (Eq. 8)

---


#### `get_true_W(fem_material,J,C,I1,I2,I3)`:

Computes the strain energy densities given the strains using the analytical description of benchmark hyperelastic material models.

_Input arguments_:

- `fem_material` - String containing the name of the benchmark hyperelastic material.
- `J` - Jacobian of Cauchy-Green deformation matrix.
- `C` - Cauchy-Green deformation matrix.
- `I1` - 1st invariant of Cauchy-Green deformation matrix.
- `I2` - 2nd invariant of Cauchy-Green deformation matrix.
- `I3` - 3rd invariant of Cauchy-Green deformation matrix.

_Output arguments_:

- `W` - Strain energy density of the specified material for the given strain.
