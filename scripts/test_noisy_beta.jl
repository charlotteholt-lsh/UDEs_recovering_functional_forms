#=====================================================================
SCRIPT TO TEST HOW WELL SYMBOLIC REGRESSION WORKS WITH NOISY FUNCTIONS
======================================================================#  

using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
using JLD2
using Random
using LinearAlgebra

# Set seed for reproducibility
rng = Random.seed!(1234)


sim_name = "noisy_beta_seeded_normalised"

#========================================================
LOAD DATA
=========================================================#

# First extract raw synthesised data
dataset = JLD2.load(datadir("synthesised_trajectories_single", "synthesised_MA.jld2"))

# Extract infectious individuals and days from the dataset
data = dataset["infectious"]
days = dataset["days"]

# Evaluate the true beta function for the noiseless data
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

true_beta_noiseless = beta0 .* exp.(-zeta .* delta .*data)



#========================================================
ADD NOISE
=========================================================#

# Taken from Pant 2025 -- need to justify this choice
noise_SD = 0.025

# add Gaussian noise
sd = noise_SD * max(0.5, maximum(true_beta_noiseless))
noise = randn(rng, length(true_beta_noiseless)) .* sd
noisy_beta = true_beta_noiseless .+ noise

# Remove negative values
noisy_beta = max.(noisy_beta, 0.0)

# Plot noisy beta against true beta
p_noise_comparison = plot(days, true_beta_noiseless, lw=2.5, label="True β (noiseless)", color=:black)
plot!(p_noise_comparison, days, noisy_beta, lw=2.5, ls=:dash, label="Noisy β", color=:red, alpha=0.8)
xlabel!(p_noise_comparison, "Day")
ylabel!(p_noise_comparison, "Transmission rate β")
title!(p_noise_comparison, "Noisy β vs True β")
#display(p_noise_comparison)

#========================================================
UNDERTAKE SYMBOIC REGRESSION
=========================================================#

# Define beta input
x_hat = reshape(data./population, :, 1)

# Define beta output
y_hat = noisy_beta 

# Create output directory
output_dir = joinpath(@__DIR__, "..", "scripts", "outputs", "noisy_beta_outputs")

model = SRRegressor(
    niterations=100,
    binary_operators=[+, -, *, /],
    unary_operators=[exp],
    maxsize = 20,
    output_directory = output_dir,
    # Make results reproducible by disabling multithreading
    parallelism=:serial,
    seed = 1234,
    deterministic = true,
    batching = false
) 


# Create and train model on this data
mach = machine(model, x_hat, y_hat)

fit!(mach)

report(mach)




#=============================================================
TESTING
=============================================================#

# Symbolic Regression result
SR_beta = predict(mach, (data=x_hat, idx=6))

mse_SR_true = Functions.loss_mse(SR_beta, true_beta_noiseless)
mse_SR_noisy = Functions.loss_mse(SR_beta, noisy_beta)

println(mse_SR_noisy)
println(mse_SR_true)

#=============================================================
PLOT RESULTS
=============================================================#

p_comparison = plot(days, true_beta_noiseless, lw=2.5, label="True β", color=:black)
plot!(p_comparison, days, SR_beta, lw=2.5, ls=:dash, label="SR-recovered β", color=:red, alpha=0.8)
plot!(p_comparison, days, vec(y_hat), lw=2.5, ls=:dot, label="Noisy β", color=:lightblue, alpha=0.8)
xlabel!(p_comparison, "Day")
ylabel!(p_comparison, "Transmission rate β")
title!(p_comparison, "Symbolic Regression vs True and Noisy β")
plot!(p_comparison, legend=:best, legendfontsize=10)
plot!(p_comparison, grid=true, gridalpha=0.3)

x_ann = days[end] * 0.75
y_ann = maximum(true_beta_noiseless) * 0.85
dy = maximum(true_beta_noiseless) * 0.06
annotate!(p_comparison, x_ann, y_ann, text("MSE (true vs SR) = $(round(mse_SR_true, sigdigits=3))", 9))
annotate!(p_comparison, x_ann, y_ann - dy, text("MSE (noisy vs SR) = $(round(mse_SR_noisy, sigdigits=3))", 9))



display(p_comparison)
figures_dir = joinpath(@__DIR__, "figures")
mkpath(figures_dir)
savefig(p_comparison, joinpath(figures_dir, "$(sim_name)_SR_beta_vs_true.png"))

