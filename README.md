# UDEs: Recovering Functional Forms

A Julia project for recovering unknown functional forms in differential equations using Universal Differential Equations (UDEs) — 
a technique that embeds neural networks within mechanistic ODE models to learn unknown dynamics from data, whilst explicitly including known dynamics of the system. 
We then aim to represent the unknown component learned by the neural network in an interpretable symbolic form via sparse regression.

Using the framework to see if UDEs are able to recover a symbolic form of behavioural response to an epidemic.

## Files

### Universal functions: functions.jl
Script containing functions that are used throughout the project.

### Generate synthetic trajectories: generate_synthetic_trajectories.jl
Script to generate synthetic trajectories of an $SEIRD$ model, with a known functional form for the transmission rate and chosen epidemiological parameters. 
Data saved as a .csv file

### Create UDE framework and train the UDE: ude_model.jl
Script to define the hyperparameters and UDE $SEIRD$ system with the transmission rate $\beta$ represented with a neural network. It loads the synthetic data for trainining, then trains the UDE. Outputs (predictions, losses, and trained neural network parameters)
saved as a .jld2 file.

### Combine multiple UDE simulations trained on the same data: model_ensemble.jl
Script to generate plots of the trained UDEs, and analyse their performance. Figures saved as a .png file.

### Undertake symbolic regression: symbolic_regression.jl
Script to recover the functional form of the behavioural response

## How to use
1. Clone the repository
2. Instantiate the Julia environment
```
using Pkg
Pkg.add("DrWatson")
Pkg.activate
Pkg.instantiate
```
3. Run the scripts
