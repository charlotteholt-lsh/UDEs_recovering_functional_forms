#=============================================================
SCRIPT TO RECOVER THE FUNCTIONAL FORM OF THE TRANSMISSION RATE
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
using .Functions
using Statistics
using Random; rng = Random.default_rng()

#=============================================================
SET UP
=============================================================#

# Define simulation name and training length
sim_name = "synthesised_MA_input_death_time_hidden_dims_5_RB_solve_no_param_check"

# Define the NN architecture
hidden_dims = 5

beta_network = Lux.Chain(Lux.Dense(2=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu),
                         Lux.Dense(hidden_dims=>1, softplus))

# Initialise parameters
p_nn_temp, st_nn = Lux.setup(rng, beta_network)

#=============================================================
CREATE BASIS
=============================================================#

# Generate library of candidate functions
# We have one state variable u[1](t)
@variables t u(t)[1:1]
u = collect(u)
# Define the library of candidate functions
h = Num[polynomial_basis(u,3); exp(u[1])]
# Define basis
basis = DataDrivenDiffEq.Basis(h, u, iv = t)


#=============================================================
RETRIEVE PREDICTIONS AND PARAMETERS FROM THE BEST SIMULATION
=============================================================#

# Load the observed data
dataset = load(datadir("sims", "synthetic_mortality_ground_truth_exp.jld2"))
# Just use data with strongest behavioural response (zeta = 0.02)
df = dataset["df"]
obs = df[!, "y_zeta_0.02"]

# Retrieve NN parameters that resulted in the lowest error on the training data

# Define the root file path
root = datadir("sims", "ude", sim_name)

# Collect results from all simulations
results_list = []
for filename in readdir(root)
    # Only include directories
    if isdir(joinpath(root, filename))
        # Extract results, predictions and losses
        SR_results = load(datadir("sims", "ude", sim_name, filename, "results.jld2"))
        pred = SR_results["prediction"]
        # Extract the predicted mortalities for the training data
        D_pred = pred[5, 1:length(obs)]
        daily_deaths_pred = [0.0; diff(D_pred)]
        mse = Functions.loss_mse(daily_deaths_pred, obs)
        push!(results_list, (mse=mse, fname=filename, daily_deaths_pred=daily_deaths_pred))
    end
end

# Find the simulation with the lowest MSE
best_idx = argmin(r.mse for r in results_list)
best_mse = results_list[best_idx].mse
best_fname = results_list[best_idx].fname

# Extract the data but convert to a 1 x N matrix
x_hat = reshape(results_list[best_idx].daily_deaths_pred, 1, :)

# Extract the NN parameters from the best simulation
best_results = load(datadir("sims", "ude", sim_name, best_fname, "results.jld2"))
p_trained = best_results["p"]
days = best_results["days"]

# Extract NN approximation
nn_input = vcat(x_hat, reshape(days, 1, :))

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
PLOT THE RESULTS
=============================================================#



#=============================================================
DEFINE INITAL STATE AND PARAMETERS
=============================================================#

# Latent period of 3 days represented by incubation rate sigma
const sigma = 1/3 
# Infectious period of 10 days represented by recovery rate gamma
const gamma = 1/10

# Massachusetts population size - taken from JHU CSSE
population = 6892503

# Define initial state same as the generated data
# Retrieve fixed parameters
E0 = 1.0
R0_recovered = 0.0
D0 = 0.0

# EXTRACTED USING GROUND_TRUTH_VALUES FOR MA IN THE PYTHON CODE
prevalence = exp(-11.468967)
R0_reproduction = 5.284404
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
    beta = nn_res([delta * I], p, t)[1]
    # Define the SEIRD equations
    du[1] = -beta * S * I / N
    du[2] = beta * S * I / N - sigma * E
    du[3] = sigma * E - (gamma + delta) * I
    du[4] = gamma * I
    du[5] = delta * I     
end

# Define ODE problem with recovered parameters form symbolic regression and solve
sindy_prob = ODEProblem(recovered_dynamics!, init_state, (1,123), nn_params)
pred_sindy = solve(sindy_prob, Tsit5(), u0=init_state, tspan=(1,123), saveat=1)

# Convert cumulative deaths D(t) to daily deaths to match observed data
daily_deaths_sindy = [0.0; diff(pred_sindy[5, :])]

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
true_prob = ODEProblem(true_dynamics!, init_state, (1,123))
pred_true = solve(true_prob, Tsit5(), u0=init_state, tspan=(1,123), saveat=1)

#=============================================================
EXTRACT THE DAILY DEATH PREDICTIONS FOR THE UDE MODEL
=============================================================#

daily_deaths_nn = results_list[best_idx].daily_deaths_pred

#=============================================================
EVALUATE BETA TRAJECTORIES
=============================================================#

# Do all relative to the same 'true' trajectory for comparison

# True beta from the true ODE trajectory (no SINDY dynamics)
I_true = pred_true[3, :]
beta_true = true_beta.(I_true)

# Beta learned by the trained NN on the same true-system I(t)
nn_input_true = vcat(reshape(delta .* I_true, 1, :), reshape(pred_true.t, 1, :))
beta_nn_on_true = vec(beta_network(nn_input_true, p_trained.nn_params, st_nn)[1])

# Recovered beta evaluated along the same I(t) for comparison
beta_recovered_on_true = [nn_res([delta * I], nn_params, t)[1] for (I, t) in zip(I_true, pred_true.t)]

# Do relative to their own trajectories for comparison

# True beta from the true ODE trajectory (no SINDY dynamics)
D_true = pred_true[5, :]
beta_true = true_beta.(D_true)

# Beta learned by the trained NN on the NN approximation
beta_nn = vec(y_hat) 

# Recovered beta evaluated along the SINDy approximation
beta_recovered = [nn_res([delta*I], nn_params, t)[1] for (I, t) in zip(pred_sindy[3, :], pred_sindy.t)]


# Plot the results against the observed data

p1 = plot(days, obs, label="Observed data", lw=2)
plot!(p1, days, daily_deaths_nn, label="NN trajectory", lw=2, ls=:dot)
plot!(p1, pred_sindy.t, daily_deaths_sindy, label="SINDY prediction", lw=2, ls=:dash)
xlabel!(p1, "Day")
ylabel!(p1, "Daily deaths")
title!(p1, "SINDY approximation of daily deaths")
mse_sindy = Functions.loss_mse(daily_deaths_sindy, obs)
mse_nn = Functions.loss_mse(daily_deaths_nn, obs)
annotate!(p1, days[end], maximum(obs), text("MSE SINDY: $(round(mse_sindy, digits=4))", 9, :right))
annotate!(p1, days[end], maximum(obs) * 0.9, text("MSE NN: $(round(mse_nn, digits=4))", 9, :right))

p2 = plot(days, beta_true, label="True β from true ODE I(t)", lw=2)
plot!(p2, days, beta_nn, label="NN-learned β on I(t) learned by NN", lw=2, ls=:dot)
plot!(p2, days, beta_recovered, label="Recovered β on recovered ODE I(t)", lw=2, ls=:dash)
xlabel!(p2, "Day")
ylabel!(p2, "β")
title!(p2, "Transmission rate using synthesised/learned/recovered I(t) respective to their own trajectories")

p3 = plot(days, beta_true, label="True β from true ODE I(t)", lw=2)
plot!(p3, days, beta_nn_on_true, label="NN-learned β on true ODE I(t)", lw=2, ls=:dot)
plot!(p3, days, beta_recovered_on_true, label="Recovered β on true ODE I(t)", lw=2, ls=:dash)
xlabel!(p3, "Day")
ylabel!(p3, "β")
title!(p3, "Transmission rate using true-system I(t)")

plot(p1, p2, p3, layout=(3, 1), size=(900, 1050))