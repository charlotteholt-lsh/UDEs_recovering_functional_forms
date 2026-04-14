#========================================================
SCRIPT TO TRAIN THE UDE MODEL
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
sim_name ="synthesised_use_normalised_infections_optimal_250326"
model_name = "ude"
if !isdir(datadir("sims", model_name, sim_name)) 
	mkpath(datadir("sims", model_name, sim_name))
end

# Number of data points used for training (total number of entries in the dataset)
const train_length = 365
const maxiters = 2500
# set the number of hidden dimensions in the neural network equal to 3
hidden_dims = 5
# do 100 simulations 
n_sims = 100

#========================================================
LOAD DATA
=========================================================#

dataset = load(datadir("synthesised_trajectories", "synthetic_pop=6892503_E0=0.0_R0=0.0_D0=0.0_sig=0.333_gam=0.1_zet=0.02_prev=1.04e-5_del=0.000131_R0r=5.28.jld2"))

# Extract infectious individuals and days from the dataset
data = dataset["infectious"]
days = dataset["days"]

#========================================================
DEFINE INITIAL STATE
=========================================================#

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
const delta = exp(-8.941224)


# Derive other parameters
beta0 = R0_reproduction * (gamma + delta)
I0 = max(1.0, prevalence * population)
S0 = population - E0 - I0 - R0_recovered - D0

# Define initial state
init_state = [S0, E0, I0, R0_recovered, D0]

#========================================================
SET UP MODEL
=========================================================#
# Do the following if we want to take weekly averages of the data (if for example there will be big trends within the week such as at weekends)

# sample_period = 7
# Define the indexes for the training data (every 7 days)
# train_split = 1:div(train_length, sample_period)
# Define the discrete timepoints where we have data for training (every 7 days)
# t_train = range(0.0, step=sample_period, length=length(train_split))
# Define the timespan for the ODE solver
tspan = [0, train_length]

# Create neural network to estimate the transmission rate:
# We have two hidden layers with hidden_dims neurons and gelu activation function
# We are taking t and I(t) as inputs and outputting beta(t)
beta_network = Lux.Chain(Lux.Dense(2=>hidden_dims, gelu), Lux.Dense(hidden_dims=>hidden_dims, gelu),
                         Lux.Dense(hidden_dims=>1, softplus))

# Initialise parameters to build the structure for the UDE
p_nn_temp, st_nn = Lux.setup(rng, beta_network)

# Convert to ComponentArray for gradient-based optimisation
# ComponentArray wraps nested parameter structures into flat array keeping named access
p_nn_temp = ComponentArray(p_nn_temp)

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
    nn_input = [I / population, t / train_length]

    # Evaluate neural network and extract scalar
    beta = beta_network(nn_input, p.nn_params, st_nn)[1][1]

    # Define the SEIRD equations
    du[1] = -beta * S * I / N
    du[2] = beta * S * I / N - sigma * E
    du[3] = sigma * E - (gamma + delta) * I
    du[4] = gamma * I
    du[5] = delta * I
end

prob_ude = ODEProblem(seird_nn!, init_state, tspan, p_nn_temp)

#========================================================
PREDICTION AND LOSS FUNCTIONS
=========================================================# 

# Predict number of daily infections (movement of people from S -> E)
function predict_ude(p_all)
    
    prob = remake(prob_ude, p = p_all)
    sol_ude = solve(prob, Rosenbrock23(), saveat=1.0, dense = false)

    I_pred = sol_ude[3, 1:train_length]

    return I_pred
end



#========================================================
TRAINING
=========================================================# 


function train_ude(p; maxiters = maxiters, halt_condition = l -> false)

    # Set up optimisation
    optimised_state = Optimisers.setup(Optimisers.Adam(1e-3), p)

    # Create 1D vector to track losses during training
    losses = Float64[]
    best_loss = Inf
    best_p = p

    for iter in 1:maxiters
        # Compute the loss, predicted mortalities and gradient function
        (l, pred), back_all = pullback(theta -> Functions.loss_ude(theta, predict_ude, data), p)
        println("Iteration $iter, Loss: $l")
        # Evaluate the gradient fo the loss w.r.t p
        grad = back_all((one(l), nothing))[1]

    	# Stop training if 5 consecutive Inf losses
		if l == Inf && length(losses) >= 5 && all(isinf, losses[end-4:end])
			println("Unstable parameter region. Aborting...")
			break
		end 

		if isnothing(grad)
			println("No gradient found. Loss: $l")
			p = best_p
			continue
		end

        push!(losses, l)

        # Store best iteration
        if l < best_loss
            best_loss = l
            best_p = p
        end

#========================================================
RUN WITHOUT PLOTTING
        if iter % 50 == 0
            display("Total loss: $l")
            x = days[1:length(pred)]
            pl = scatter(x, data[1:length(pred)], color=:black, markersize=2,
                label="Data", xlabel="Day", ylabel="Daily new infections", title="Iteration $iter")
            plot!(pl, x, pred, color=:red, linewidth=2, label="Prediction")
            display(pl)
		end	
========================================================#


        # Update parameters using the gradient
        optimised_state, p = Optimisers.update(optimised_state, p, grad)

        if halt_condition(l)
			break
		end
    end

    return best_p, losses
end

#========================================================
MAIN FUNCTION TO TRAIN THE UDE AND SAVE THE RESULTS
=========================================================# 

function run_model()
    println("Starting run: on thread $(Threads.threadid())")

    # Initialise parameters
    p, st = Lux.setup(rng, beta_network)
    p = ComponentArray(p)


    # Combine all parameters into a single object for optimisation
    p_init = ComponentArray(
        nn_params = p,
        gamma = gamma,
        sigma = sigma,
        delta = delta,
        tmax = train_length
    )


    # Make sure to start with a stable parameterization
    l_init = Functions.loss_ude(p_init, predict_ude, data)[1]
    println("Initial loss: $l_init")
#========================================================
    while l_init > 1e4
		println("Unstable initial parameterization. Restarting..., $l_init")
        # Initialise parameters
        p, st = Lux.setup(rng, beta_network)
        p = ComponentArray(p)


        # Combine all parameters into a single object for optimisation
        p_init = ComponentArray(
            nn_params = p,
            gamma = gamma,
            sigma = sigma,
            delta = delta,
            tmax = train_length
        )
        l_init = Functions.loss_ude(p_init, predict_ude, data)[1]
	end
=========================================================# 

    halt_condition = l -> (abs(l) < 0.01)
    p_trained, losses_final = train_ude(p_init, maxiters = maxiters, halt_condition = halt_condition)

    # Evaluate final long term results 
    long_term_prob= remake(prob_ude, p = p_trained, tspan = (0.0, 3*365.0))
    long_term_pred = solve(long_term_prob, Rosenbrock23(), saveat=1, dense = false)

    region = "Massachusetts"
    param_name = hidden_dims

	# Save the result
	fname = "$(region)_$(param_name)_t$(Threads.threadid())"

	# Append a number ot the end of the simulation to allow multiple runs of a single set of hyperparameters for ensemble predictions
	model_iteration = 1
	while isdir(datadir("sims", model_name, sim_name, "$(fname)_v$(model_iteration)"))
		model_iteration += 1
	end
	fname = fname * "_v$model_iteration"

	mkpath(datadir("sims", model_name, sim_name, fname))


	save(datadir("sims", model_name, sim_name, fname, "results.jld2"),
		"p", p_trained, "losses", losses_final, "prediction", Array(long_term_pred),
		"days", days)
	println("Finished run: $(region) on thread $(Threads.threadid())")

	return nothing
end

for i = 1:n_sims
    run_model()
end

