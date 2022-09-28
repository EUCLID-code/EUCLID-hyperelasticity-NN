# train.py

This file contains scripts to train the ICNN using the physics-guided loss function.

#### `train_weak(model, datasets, fem_material, noise_level)`:


_Input arguments_:

- `model` - class instance of the ICNN model that is to be trained.
- `datasets` - dataset to be used to train the ICNN model.
- `fem_material` - name of the material to be learned (for file naming convention).
- `noise_level` - level of noise used to condition the dataset (for file naming convention).

_Output arguments_:

- `model` - returns trained model class instance
- `loss_history` - returns loss history
