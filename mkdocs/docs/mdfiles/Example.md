# Example

To train the ICNN to learn the hidden Isihara model from the dataset with high noise conditioning run the following:

- `python main.py Isihara high`


The script will train a number of neural networks (specified in `config.py`) and evaluate each model against the ground-truth model along six deformation paths. The models will be saved in `results/Isihara/`. A plot will be saved in `results/` including strain energy density and Piola Kirchhoff stress tensor component evaluations along the deformation paths.
