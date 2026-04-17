#=============================================================
SCRIPT TO RECOVER THE FUNCTIONAL FORM OF THE TRANSMISSION RATE FOR A SINGLE TRAJECTORY
=============================================================#

using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using DataDrivenSparse
using LinearAlgebra
using Lux
using JLD2
using ComponentArrays
using Plots
include(joinpath(@__DIR__, "functions.jl"))
using .Functions
using Statistics
using Random; rng = Random.default_rng()

#=============================================================
SET UP
=============================================================#

# Define simulation name and training length
sim_name = "synthesised_use_normalised_infections_optimal_250326"

# Define the NN architecture
hidden_dims = 5

beta_network = Lux.Chain(Lux.Dense(2=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu),
                         Lux.Dense(hidden_dims=>1, softplus))

# Initialise parameters
p_nn_temp, st_nn = Lux.setup(rng, beta_network)

# Define population for scaling
population = 6892503.0

#=============================================================
CREATE BASIS
=============================================================#

# Generate library of candidate functions
# We have one state variable u[1](t)
@variables t u(t)[1:1]
# Normalise the number of infectious individuals so that the exponential term doesn't overflow
u_scaled = u[1] / population
poly_terms = DataDrivenDiffEq.polynomial_basis([u_scaled], 3)
h = Num[vcat(poly_terms, [exp(u_scaled)])...]
# Define basis
basis = DataDrivenDiffEq.Basis(h, u, iv = t)


#=============================================================
RETRIEVE PREDICTIONS AND PARAMETERS FROM THE BEST SIMULATION
=============================================================#

# Load the observed data
dataset = load(datadir("synthesised_trajectories_old", "synthetic_pop=6892503_E0=0.0_R0=0.0_D0=0.0_sig=0.333_gam=0.1_zet=0.02_prev=1.04e-5_del=0.000131_R0r=5.28.jld2"))
   
# Just use infectious trajectory
obs = dataset["infectious"]
days = dataset["days"]

# Retrieve NN parameters that resulted in the lowest error on the training data

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
x_hat = reshape(results_list[best_idx].i_traj, 1, :)

# Extract the NN parameters from the best simulation
best_results = load(datadir("sims", "ude_single", sim_name, best_fname, "results.jld2"))
p_trained = best_results["p"]
days = best_results["days"]

# Define training length for normalization (must match ude_model_single_traj.jl)
const train_length = 365

# Extract NN approximation (normalised)
nn_input = vcat(x_hat ./ population, reshape(days ./ train_length, 1, :))


# Evaluate neural network and extract approximation
y_hat = beta_network(nn_input, p_trained.nn_params, st_nn)[1]


#=============================================================
DEFINE AND SOLVE THE SPARSE REGRESSION PROBLEM
=============================================================#

nn_problem = DataDrivenDiffEq.DirectDataDrivenProblem(x_hat, y_hat)

# Use STLSQ to solve the sparse regression problem
# Define the shrinking cut off
lambda = 1e-4
opt = DataDrivenSparse.STLSQ(lambda)

# lambdas = exp10.(-3:0.1:3)
# opt = STLSQ(lambdas)

# Solve the sparse regression problem
options = DataDrivenDiffEq.DataDrivenCommonOptions(maxiters = 10_000,
          normalize = DataDrivenDiffEq.DataNormalization(DataDrivenDiffEq.ZScoreTransform),
          selector = DataDrivenDiffEq.bic, digits = 1,
          data_processing = DataDrivenDiffEq.DataProcessing(split = 0.9,
          batchsize = 30,
          shuffle = true,
          rng = rng))

nn_res = DataDrivenDiffEq.solve(nn_problem, basis, opt, options=options)
# nn_res = DataDrivenDiffEq.solve(nn_problem, basis, opt, progress=true, normalize=true, denoise=false)

nn_eqs = DataDrivenDiffEq.get_basis(nn_res)
nn_params = DataDrivenDiffEq.get_parameter_values(nn_eqs)
println(nn_res)
println(nn_eqs)
println(nn_params)

#=============================================================
DEFINE INITAL STATE AND PARAMETERS
=============================================================#

# Define training length
const train_length = 365

# Latent period of 3 days represented by incubation rate sigma
const sigma = 1/3 
# Infectious period of 10 days represented by recovery rate gamma
const gamma = 1/10

# Massachusetts population size - taken from JHU CSSE
population = 6892503

# Define initial state same as the generated data
# Retrieve fixed parameters
const E0 = 1.0
const R0_recovered = 0.0
const D0 = 0.0

# EXTRACTED USING GROUND_TRUTH_VALUES FOR MA IN THE PYTHON CODE
const prevalence = exp(-11.468967)
const R0_reproduction = 5.284404
const zeta = 0.02
const delta = exp(-8.941224)


# Derive other parameters
beta0 = R0_reproduction * (gamma + delta)
I0 = max(1.0, prevalence * population)
S0 = population - E0 - I0 - R0_recovered - D0

# Define initial state
init_state = [S0, E0, I0, R0_recovered, D0]

#=============================================================
DEFINE ODE SYSTEM USING THE BETA THAT HAS BEEN RECOVERED FROM THE NN BY SYMBOLIC REGRESSION
=============================================================#

function recovered_dynamics!(du, u, p, t)
    S, E, I, R, D = u
    # Define the population size
    N = S + E + I + R
    # If population size less than or equal to zero return zero
    if N <= 0
        du .= 0.0
        return
    end
    # Evaluate symbolic-regression approximation
    beta = nn_res([I], p, t)[1]
    # Define the SEIRD equations
    du[1] = -beta * S * I / N
    du[2] = beta * S * I / N - sigma * E
    du[3] = sigma * E - (gamma + delta) * I
    du[4] = gamma * I
    du[5] = delta * I     
end

# Define ODE problem with recovered parameters form symbolic regression and solve
sindy_prob = ODEProblem(recovered_dynamics!, init_state, (1,train_length), nn_params)
pred_sindy = solve(sindy_prob, Tsit5(), u0=init_state, tspan=(1,train_length), saveat=1)

#=============================================================
DEFINE ODE SYSTEM USING THE KNOWN FUNCTIONAL FORM FOR THE TIME-VARYING BETA
=============================================================#

# Define known beta function
function true_beta(I)
    arg_exp = clamp(zeta * delta * I, -50.0, 50.0)
    beta = beta0 * exp(-arg_exp)
    return beta
end

# Define dynamics using the known beta function
function true_dynamics!(du, u, p, t)
    S, E, I, R, D = u
    # Define the population size
    N = S + E + I + R
    # If population size less than or equal to zero return zero
    if N <= 0
        du .= 0.0
        return
    end
    # Evaluate true transmission function
    beta = true_beta(I)
    # Define the SEIRD equations
    du[1] = -beta * S * I / N
    du[2] = beta * S * I / N - sigma * E
    du[3] = sigma * E - (gamma + delta) * I
    du[4] = gamma * I
    du[5] = delta * I
end

# Define and solve ODE problem using parameters defined at the start of the script
true_prob = ODEProblem(true_dynamics!, init_state, (1,train_length))
pred_true = solve(true_prob, Tsit5(), u0=init_state, tspan=(1,train_length), saveat=1)

#=============================================================
EXTRACT THE DAILY DEATH PREDICTIONS FOR THE UDE MODEL
=============================================================#

i_traj_nn = results_list[best_idx].i_traj

#=============================================================
EVALUATE BETA TRAJECTORIES
=============================================================#

# Do all relative to the same 'true' trajectory for comparison

# True beta from the true ODE trajectory (no SINDY dynamics)
I_true = pred_true[3, :]
beta_true = true_beta.(I_true)

# Beta learned by the trained NN on the same true-system I(t)
nn_input_true = vcat(reshape(I_true, 1, :)./population, reshape(pred_true.t, 1, :)./train_length)
beta_nn_on_true = vec(beta_network(nn_input_true, p_trained.nn_params, st_nn)[1])

# Recovered beta evaluated along the same I(t) for comparison
beta_recovered_on_true = [nn_res([I], nn_params, t)[1] for (I, t) in zip(I_true, pred_true.t)]

# Do relative to their own trajectories for comparison

# Beta learned by the trained NN on the NN approximation
beta_nn = vec(y_hat) 

# Recovered beta evaluated along the SINDy approximation
beta_recovered = [nn_res([I], nn_params, t)[1] for (I, t) in zip(pred_sindy[3, :], pred_sindy.t)]

#=============================================================
PLOT RESULTS
=============================================================#

# Plot predicted trajectories (UDE and SINDy model) against the observed data
p1 = plot(days, obs, label="Observed data", lw=2)
plot!(p1, days, i_traj_nn, label="NN trajectory", lw=2, ls=:dot)
plot!(p1, pred_sindy.t, pred_sindy[3, :], label="SINDY prediction", lw=2, ls=:dash)
xlabel!(p1, "Day")
ylabel!(p1, "Daily deaths")
title!(p1, "SINDY approximation of daily deaths")
mse_sindy = Functions.loss_mse(pred_sindy[3, :], obs)
mse_nn = Functions.loss_mse(i_traj_nn, obs)
annotate!(p1, days[end], maximum(obs), text("MSE SINDY: $(round(mse_sindy, digits=4))", 9, :right))
annotate!(p1, days[end], maximum(obs) * 0.9, text("MSE NN: $(round(mse_nn, digits=4))", 9, :right))

# Plot the beta trajectories evaluated against their respective predicted/observed trajectories
p2 = plot(days, beta_true, label="True β from true ODE I(t)", lw=2)
plot!(p2, days, beta_nn, label="NN-learned β on I(t) learned by NN", lw=2, ls=:dot)
plot!(p2, days, beta_recovered, label="Recovered β on recovered ODE I(t)", lw=2, ls=:dash)
xlabel!(p2, "Day")
ylabel!(p2, "β")
title!(p2, "Transmission rate using synthesised/learned/recovered I(t) respective to their own trajectories")

# Plot the beta trajectories evaluated against the observed trajectory
p3 = plot(days, beta_true, label="True β from true ODE I(t)", lw=2)
plot!(p3, days, beta_nn_on_true, label="NN-learned β on true ODE I(t)", lw=2, ls=:dot)
plot!(p3, days, beta_recovered_on_true, label="Recovered β on true ODE I(t)", lw=2, ls=:dash)
xlabel!(p3, "Day")
ylabel!(p3, "β")
title!(p3, "Transmission rate using true-system I(t)")

pl = plot(p1, p2, p3, layout=(3, 1), size=(900, 1050))

# Save the plot
savefig(pl, joinpath(@__DIR__, "figures", "$(sim_name)_sindy_trajectory.png"))           

display(pl)

p_beta = plot(days, beta_true, label = "True β", lw = 2)
plot!(p_beta, days, beta_nn_on_true, label = "Learned β", lw = 2, ls = :dash)

xlabel!(p_beta, "Day")
ylabel!(p_beta, "β")
title!(p_beta, "Learned vs true transmission rate")

display(p_beta)

savefig(p_beta, joinpath(@__DIR__, "figures", "$(sim_name)_sindy_beta_comparison.png"))