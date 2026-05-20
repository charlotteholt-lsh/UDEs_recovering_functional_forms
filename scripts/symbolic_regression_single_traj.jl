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
using Plots
using Statistics
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
using .EstimatedGroundTruthParameters: POPULATION, PREVALENCE, R0_REPRODUCTION, DELTA, ZETA

location = "MA"
population = POPULATION[location]
prevalence = PREVALENCE[location]
delta = DELTA[location]
R0_reproduction = R0_REPRODUCTION[location]
zeta = ZETA[location]

# Derive other parameters
# Infectious period of 10 days represented by recovery rate gamma
const gamma = 1/10
beta0 = R0_reproduction * (gamma + delta)

# Retrieve NN parameters that resulted in the lowest error on the training data

# Load the observed data
dataset = JLD2.load(datadir("synthesised_trajectories_single", "synthesised_MA.jld2"))
   
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
        SR_results = JLD2.load(datadir("sims", "ude_single", sim_name, filename, "results.jld2"))
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
best_results = JLD2.load(datadir("sims", "ude_single", sim_name, best_fname, "results.jld2"))
p_trained = best_results["p"]
days = best_results["days"]

# Extract NN approximation (normalised)
nn_input = I_nn ./ population

x_hat = permutedims(nn_input)

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

#=============================================================
TESTING
=============================================================#

# Symbolic Regression result
SR_beta = exp.(obs ./ population .* -5.3842466426660724) .* 0.5319183166047315

# True beta result
true_beta_SR = beta0 .* exp.(-zeta .* delta .*obs)

mse_SR_true = Functions.loss_mse(SR_beta, true_beta_SR)
mse_SR_NN = Functions.loss_mse(SR_beta, vec(y_hat))

println(beta0)
println(zeta*delta*population)

diff_coeff = beta0 - 0.5319183166047315
diff_exp_coeff = (zeta * delta) - (-5.3842466426660724)/population
println("Difference in coefficient: $diff_coeff")
println("Difference in exponent coefficient: $diff_exp_coeff")
#=============================================================
PLOT RESULTS
=============================================================#

p_comparison = plot(days, true_beta_SR, lw=2.5, label="True β", color=:black)
plot!(p_comparison, days, SR_beta, lw=2.5, ls=:dash, label="SR-recovered β", color=:red, alpha=0.8)
plot!(p_comparison, days, vec(y_hat), lw=2.5, ls=:dot, label="NN approximation β", color=:lightblue, alpha=0.8)
xlabel!(p_comparison, "Day")
ylabel!(p_comparison, "Transmission rate β")
title!(p_comparison, "Symbolic Regression vs True β")
plot!(p_comparison, legend=:best, legendfontsize=10)
plot!(p_comparison, grid=true, gridalpha=0.3)

x_ann = days[end] * 0.75
y_ann = maximum(true_beta_SR) * 0.85
dy = maximum(true_beta_SR) * 0.06
annotate!(p_comparison, x_ann, y_ann, text("MSE (true vs SR) = $(round(mse_SR_true, sigdigits=3))", 9))
annotate!(p_comparison, x_ann, y_ann - dy, text("MSE (SR vs NN) = $(round(mse_SR_NN, sigdigits=3))", 9))



display(p_comparison)
figures_dir = joinpath(@__DIR__, "figures")
mkpath(figures_dir)
savefig(p_comparison, joinpath(figures_dir, "$(sim_name)_SR_beta_vs_true.png"))

