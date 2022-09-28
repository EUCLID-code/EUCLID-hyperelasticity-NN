# main.py

This is the main file to train the ICNN-based material model.

The arguments to run the file are:

- `<fem_material>` - can be any one of the following: `NeoHookean`, `Isihara`, `HainesWilson`, `GentThomas`, `ArrudaBoyce`, `Ogden`, `Anisotropy45`, `Anisotropy60`, `Holzapfel`
- `<noise_material>` - noise conditioning of the data (can be `low` or `high`)

The individual components of the main file are the following:

- Reads the command line arguments `<fem_material>` and `<noise_material>` and loads the datasets accordingly.
- Initializes ICNN model and assigns randomly sampled weights and biases to it (sampled via `xavier_uniform`)
- Trains a number of ICNNs (equal to `ensemble_size` defined in `config.py`) each with randomly initialized weights.
- Saves each model to the output directory defined in `config.py`.
- Once all models are trained `evaluate_icnn()` is called to evaluate each ICNN in the ensemble and plot the performance against the ground truth model along six deformation paths.
