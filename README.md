# deep_ensemble_flux_inference
 
This code can be used to disaggregate and gap-fill eddy covariance flux measurements based on a set of predictors, weights of surface classes, and total flux estimates using an ensemble of Baysian neural networks.

For eddy covariance data, the weights of surface classes could be estimated by a surface class map combined with a flux footprint model, e.g., https://footprint.kljun.net/index.php

Use the infer_fluxes.py script to run the analysis and plot the results (.TXT file) with your favorite toolbox.

The given example resembles the analysis in https://essopenarchive.org/doi/full/10.22541/essoar.168394762.23256034/v1
The dataset is provided at doi.org/10.5281/zenodo.7913027
The flux inference assumes 3 surface classes (called Ponds, Palsas, and Fens) and performs the disaggregation for CO2 and CH4 fluxes seperately.
