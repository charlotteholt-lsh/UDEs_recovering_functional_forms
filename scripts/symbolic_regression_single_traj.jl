#========================================================
SCRIPT TO USE SYMBOLICREGRESSION.JL FOR SYMBOLIC REGRESSION
=========================================================#  

using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
using JLD2
# Import symbolic regression package and MLJ interface
using SymbolicRegression
using Lux
using MLJ 
using Random; rng = Random.default_rng()
# Call the loss functions
include(joinpath(@__DIR__, "functions.jl"))
using .Functions

#========================================================
SET UP NEURAL NETWORK
=========================================================#  

# Define the NN architecture
hidden_dims = 5

# Retrieve nn architecture
beta_network = Lux.Chain(Lux.Dense(1=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu),
                         Lux.Dense(hidden_dims=>1, softplus))

# Initialise parameters
p_nn_temp, st_nn = Lux.setup(rng, beta_network)


#=============================================================
RETRIEVE PREDICTIONS AND PARAMETERS FROM THE BEST SIMULATION
=============================================================#

sim_name = "270426_post_nina_comments"


# Extract from estimated ground truths
include("estimated_ground_truth_parameters.jl")
using .EstimatedGroundTruthParameters: POPULATION

location = "MA"
population = POPULATION[location]


# Retrieve NN parameters that resulted in the lowest error on the training data

# Load the observed data
dataset = load(datadir("synthesised_trajectories_single", "synthesised_MA.jld2"))
   
# Just use infectious trajectory
obs = dataset["infectious"]
days = dataset["days"]

# Define the root file path
root = datadir("sims", "ude_single", sim_name)

# Collect results from all simulations
results_list = []
for filename in readdir(root)
    # Only include directories
    if isdir(joinpath(root, filename))
        # Extract results, predictions and losses
        SR_results = load(datadir("sims", "ude_single", sim_name, filename, "results.jld2"))
        pred = SR_results["prediction"]
        # Extract the predicted infectious trajectory for the training data
        i_traj = pred[3, 1:length(obs)]
        mse = Functions.loss_mse(i_traj, obs)
        push!(results_list, (mse=mse, fname=filename, i_traj=i_traj))
    end
end

# Find the simulation with the lowest MSE
best_idx = argmin(r.mse for r in results_list)
best_mse = results_list[best_idx].mse
best_fname = results_list[best_idx].fname

# Extract the data but convert to a 1 x N matrix
I_nn = reshape(results_list[best_idx].i_traj, 1, :)

# Extract the NN parameters from the best simulation
best_results = load(datadir("sims", "ude_single", sim_name, best_fname, "results.jld2"))
p_trained = best_results["p"]
days = best_results["days"]

# Extract NN approximation (normalised)
nn_input = I_nn ./ population

x_hat =permutedims(nn_input)
# Evaluate neural network and extract approximation
y_hat = vec(beta_network(nn_input, p_trained.nn_params, st_nn)[1])

#=============================================================
UNDERTAKE SYMBOLIC REGRESSION IN MLJ
=============================================================#

model = SRRegressor(
    niterations=100,
    binary_operators=[+, -, *, /],
    unary_operators=[exp],
    maxsize = 20
)

# Create and train model on this data
mach = machine(model, x_hat, y_hat)
fit!(mach)

report(mach)