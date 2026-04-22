#========================================================
SCRIPT TO TRAIN THE UDE MODEL FOR MULTIPLE TRAJECTORIES
=========================================================#  
using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
using Lux
using ComponentArrays
using DataFrames
using DiffEqFlux, Zygote
using Optimisers
using DifferentialEquations
using Plots
using Random; rng = Random.default_rng()
# Call the loss functions
include(joinpath(@__DIR__, "functions.jl"))
using .Functions

#========================================================
DEFINE HYPERPARAMETERS
=========================================================#

# Define strings for file names and directory for results
sim_name ="synthesised_use_5_inputs_optimal_250326"
model_name = "ude_multiple"
if !isdir(datadir("sims", model_name, sim_name)) 
	mkpath(datadir("sims", model_name, sim_name))
end

# Number of data points used for training (total number of entries in the dataset)
const train_length = 365
const maxiters = 2500
# set the number of hidden dimensions in the neural network equal to 3
hidden_dims = 5

#========================================================
DEFINE INITIAL STATE
=========================================================#

# Latent period of 3 days represented by incubation rate sigma
const sigma = 1/3 
# Infectious period of 10 days represented by recovery rate gamma
const gamma = 1/10

# Define initial state same as the generated data
# Retrieve fixed parameters
const E0 = 1.0
const R0_recovered = 0.0
const D0 = 0.0


#========================================================
SET UP MODEL
=========================================================#

# Define the timespan for the ODE solver
tspan = [0, train_length]

# Create neural network to estimate the transmission rate:
# We have two hidden layers with hidden_dims neurons and gelu activation function
# We are taking beta0, zeta, dleta, I(t) and t as inputs and outputting beta(t)
beta_network = Lux.Chain(Lux.Dense(5=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu),
                         Lux.Dense(hidden_dims=>1, softplus))

# Initialise placeholder parameters to build the structure for the UDE
p_nn_temp, st_nn = Lux.setup(rng, beta_network)
init_state = [0.0, E0, 0.0, R0_recovered, D0]

# Convert to ComponentArray for gradient-based optimisation
# ComponentArray wraps nested parameter structures into flat array keeping named access
p_nn_temp = ComponentArray(p_nn_temp)


# Create placeholder component array to set up neural network
ph_nn = ComponentArray(
                nn_params = p_nn_temp,
                population = 0,
                prevalence = 0,
                beta0 = 0,
                zeta = 0,
                r0_reproduction = 0,
                delta = 0
            )

# Define the model 
function seird_nn!(du, u, p, t)
    S, E, I, R, D = u

    # Define the population size
    N = S + E + I + R

    # If population size less than or equal to zero return zero
    if N <= 0
        du .= 0.0
        return
    end

    # Define normalised inputs for NN
    nn_input = [p.beta0, p.zeta, p.delta, I / p.population, t / train_length]

    # Evaluate neural network and extract scalar
    beta = beta_network(nn_input, p.nn_params, st_nn)[1][1]

    # Define the SEIRD equations
    du[1] = -beta * S * I / N
    du[2] = beta * S * I / N - sigma * E
    du[3] = sigma * E - (gamma + p.delta) * I
    du[4] = gamma * I
    du[5] = p.delta * I
end

prob_ude = ODEProblem(seird_nn!, init_state, tspan, p_nn_temp)

#========================================================
PREDICTION AND LOSS FUNCTIONS
=========================================================# 

# Predict number of daily infections (movement of people from S -> E)
function predict_ude(p_all)
    
    # Define initial state
    I0 = max(1.0, p_all.prevalence * p_all.population)
    S0 = p_all.population - E0 - I0 - R0_recovered - D0
    init_state = [S0, E0, I0, R0_recovered, D0]

    prob = remake(prob_ude, u0 = init_state, p = p_all)
    sol_ude = solve(prob, Rosenbrock23(), saveat=1.0, dense = false)

    I_pred = sol_ude[3, 1:train_length]

    return I_pred
end

# Loss function in functions.jl module

#========================================================
TRAINING
=========================================================# 

function train_ude(nn_params; maxiters = maxiters)

    # Set up optimisation
    optimised_state = Optimisers.setup(Optimisers.Adam(1e-3), nn_params)

    # Preload all trajectories and store in a vector
    root = datadir("synthesised_trajectories")
    trajectories = []
    for filename in readdir(root)
        if endswith(filename, ".jld2")
            dataset = load(joinpath(root, filename))
            varying_p = ComponentArray(
                population = dataset["varying_p"]["population"],
                prevalence = dataset["varying_p"]["prevalence"],
                delta = dataset["varying_p"]["delta"],
                R0_reproduction = dataset["varying_p"]["R0_reproduction"],
                zeta = dataset["varying_p"]["zeta"]
            )

            push!(trajectories, (
                filename = filename,
                data = dataset["infectious"],
                days = dataset["days"],
                varying_p = varying_p
            ))
        end
    end
    println("Loaded $(length(trajectories)) trajectories into memory")

    # Create 1D vector to track total losses across all trajectories during training
    total_losses = Float64[]
    best_loss = Inf

    # We track the best parameters for all trajectories
    # the relationship between the parameters (e.g. NN weights) is the same across trajectories, but the NN will have different inputs
    best_nn_params = nn_params

    for iter in 1:maxiters
        # Compute the loss, predicted mortalities and gradient function for each synthesised trajectory
        
        # Loop through all simulations
        individual_losses = Float64[]
        # Sum the gradients with respect to the NN parameters only
        grad_sum = zero(nn_params)

        for traj in trajectories
            data = traj.data
            days = traj.days
            varying_p = traj.varying_p

            # Derive beta0 specific to current trajectory
            beta0 = varying_p.R0_reproduction * (gamma + varying_p.delta)

            # Update the parameters for the current trajectory to include the varying parameters
            p_all = ComponentArray(
                nn_params = nn_params,
                population = varying_p.population,
                prevalence = varying_p.prevalence,
                beta0 = beta0,
                zeta = varying_p.zeta,
                r0_reproduction = varying_p.R0_reproduction,
                delta = varying_p.delta
            )

            # Compute the loss and gradient for the current trajectory
            (l, pred), back_all = pullback(theta -> Functions.loss_ude(theta, predict_ude, data), p_all)
            println("Iteration $iter, Loss: $l")

            # Evaluate the gradient of the loss for the current trajectory w.r.t p_all
            grad = back_all((one(l), nothing))[1]
            
            push!(individual_losses, l)

            grad_sum .+= grad.nn_params

        end

        # Sum the losses across all trajectories to get the total loss for this iteration
        total_loss = sum(individual_losses)
        total_grad = grad_sum 

    	# Stop training if 5 consecutive Inf losses
		if total_loss == Inf && length(total_losses) >= 4 && all(isinf, total_losses[end-4:end])
			println("Unstable parameter region. Aborting...")
			break
		end 

		if isnothing(total_grad)
			println("No gradient found. Loss: $total_loss")
			nn_params = best_nn_params
			continue
		end   

        push!(total_losses, total_loss)

        # Store best iteration
        if total_loss < best_loss
            best_loss = total_loss
            best_nn_params = nn_params
        end

        # Update parameters using the gradient
        optimised_state, nn_params = Optimisers.update(optimised_state, nn_params, total_grad)

    end

    return best_nn_params, total_losses
end

#========================================================
MAIN FUNCTION TO TRAIN THE UDE AND SAVE THE RESULTS
=========================================================# 

function run_model()
    println("Starting run: on thread $(Threads.threadid())")

    # Initialise parameters
    nn_params, st = Lux.setup(rng, beta_network)
    nn_params = ComponentArray(nn_params)

    p_trained, losses_final = train_ude(nn_params, maxiters = maxiters)

    # Save the trained parameters and losses for the combined trajectories

    # Create numbered simulation folders to allow multiple runs of a single set of hyperparameters 
    model_iteration = 1
    while isdir(datadir("sims", model_name, sim_name, "simulation_v$(model_iteration)"))
        model_iteration += 1
    end
    foldername = "simulation_v$model_iteration"

    save(datadir("sims", model_name, sim_name, foldername, "training_results.jld2"), 
    "p_trained", p_trained, "losses_final", losses_final)

    # Evaluate the trained model on each trajectory and save the results
    # Loop through all simulations
    # Define the root file path
    root = datadir("synthesised_trajectories")
    # Read all files/folders in the root directory
    for filename in readdir(root)
        if endswith(filename, ".jld2")
            # Extract trajectory of infectious individuals
            dataset = load(datadir("synthesised_trajectories", filename))
            data = dataset["infectious"]
            days = dataset["days"]

            varying_p = ComponentArray(
                population = dataset["varying_p"]["population"],
                prevalence = dataset["varying_p"]["prevalence"],
                delta = dataset["varying_p"]["delta"],
                R0_reproduction = dataset["varying_p"]["R0_reproduction"],
                zeta = dataset["varying_p"]["zeta"]
            )

            # Derive beta0 specific to current trajectory
            beta0 = varying_p.R0_reproduction * (gamma + varying_p.delta)

            # Update the parameters for the current trajectory to include the varying parameters
            p_all = ComponentArray(
                nn_params = p_trained,
                population = varying_p.population,
                prevalence = varying_p.prevalence,
                beta0 = beta0,
                zeta = varying_p.zeta,
                R0_reproduction = varying_p.R0_reproduction,
                delta = varying_p.delta
            )

            # Define initial state
            I0 = max(1.0, p_all.prevalence * p_all.population)
            S0 = p_all.population - E0 - I0 - R0_recovered - D0
            init_state = [S0, E0, I0, R0_recovered, D0]

            # Evaluate prediction for the trained parameters on the current trajectory
            long_term_prob= remake(prob_ude, u0 = init_state, p = p_all)
            long_term_pred = solve(long_term_prob, Rosenbrock23(), saveat=1, dense = false)
            
            # Convert to a 1 x N matrix
            x_hat = reshape(long_term_pred[3, 1:length(data)], 1, :)

            # Define the neural network input for the current trajectory 
            # Keep beta, zeta, delta constant over time
            nn_input = vcat(fill(p_all.beta0, 1, length(days)), fill(p_all.zeta, 1, length(days)), 
                fill(p_all.delta, 1, length(days)), reshape(x_hat ./ p_all.population, 1, :), reshape(days ./ train_length, 1, :))

            # Evaluate neural network and extract approximation
            beta_traj = beta_network(nn_input, p_trained, st_nn)[1]

            # Within this folder create a folder for each trajectory
            if !isdir(datadir("sims", model_name, sim_name, foldername, filename)) 
                mkpath(datadir("sims", model_name, sim_name, foldername, filename))
            end

            # In this folder save the infectious trajectory results and the beta results for this trajectory
            save(datadir("sims", model_name, sim_name, foldername, filename, "results.jld2"), 
                "infectious_traj_prediction", Array(long_term_pred), "beta_traj", beta_traj,
                "days", days)
        end
    end
    println("Finished run: $(foldername) on thread $(Threads.threadid())")
	return nothing
end

for i = 1:1
    run_model()
end


