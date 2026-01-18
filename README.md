# FRB Dispersion Measure Predictor

This project uses Deep Learning to predict the Dispersion Measure (DM) of Fast Radio Bursts (FRBs) from raw telescope spectrograms.

Fast Radio Bursts (FRBs) are millisecond-duration radio transients of extragalactic origin.
A key physical parameter used to characterize FRBs is the Dispersion Measure (DM), traditionally estimated using computationally expensive brute-force de-dispersion techniques.

The core innovation of this model is a "Sim-to-Real" workflow. Because real telescope data is noisy and scarce, the model is trained entirely on Physics-Aligned Synthetic Data generated with realistic artifacts (RFI stripes, scattering, and channel noise). By learning to read dirty synthetic signals, the model successfully generalizes to predict DM values from real CHIME telescope observations with high accuracy.

I will be adding the comments to most of the files at the end of the project and try to explain the astronomical concepts and formuals required to understand and complete the project.

## Final Results

Below is the evaluation on real telescope data, showing the correlation between the True DM and the Predicted DM.

![Final Evaluation Plot](final_results.png)