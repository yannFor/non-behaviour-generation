# Automatic facial expressions, gaze direction and head movements generation of a virtual agent

The code contains two diffusion models to jointly and automatically generate the head, facial and gaze movements  of a virtual agent from acoustic speech features. Two architectures are explored: a Diffusion Model which integrate the audio condition with attention and a Diffusion Model which integrate the audio condition with concatenation. Head movements and gaze orientation are generated as 3D coordinates, while facial expressions are generated using action units based on the facial action coding system.

## To reproduce
1. Clone the repository
2. In a conda console, execute 'conda env create -f environment.yml' to create the right conda environment. Go to the project location.