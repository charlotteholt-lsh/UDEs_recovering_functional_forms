
#========================================================
SCRIPT TO GENERATE SYNTHETIC TRAJECTORIES
=========================================================# 

using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)

using DrWatson

# Packages required
using CSV
using JLD2
using DataFrames

# Define parameters

# Massachusetts population size - taken from JHU CSSE
N = 6892503

# Initial conditions
E0 = 0.0
R0 = 0.0
D0 = 0.0


# Latent period of 3 days represented by incubation rate sigma
const sigma = 1/3 
# Infectious period of 10 days represented by recovery rate gamma
const gamma = 1/10

#========================================================
HELPER FUNCTIONS
=========================================================#

function moving_average(x)
    k = 7
    kernel = ones(k) / k
    if length(x) < k
        return zeros(length(x))
    end
    valid = conv(x, kernel)
    return vcat(zeros(k-1), valid)
end

function extract_param_values(beta_form, location)
    # Only undertake for MA
    dt_location = "MA"
    model_name = "exp"
    RESULT = 1
    posterior_path = f"../results/{gt_location}/{model_name}_posterior_{RESULT}.csv"
end


#========================================================
DEFINE THE MODEL
=========================================================#

# Define the model to generate data with a functional form for beta
# p must be of the form (beta0, delta, sigma, gamma, zeta)
function seird_functional!(du, u, p, t)
    S, E, I, R, D = u

    # Define the population size
    N = S + E + I + R

    # If population size less than or equal to zero return zero
    if N <= 0
        du .= 0.0
        return
    end

    # Define the functional form of beta
    arg_exp = clamp(p.zeta * p.delta * I, -50.0, 50.0)
    beta = p.beta0 * exp(-arg_exp)

    # Define the SEIRD equations
    du[1] = -beta * S * I / N
    du[2] = beta * S * I / N - p.sigma * E
    du[3] = p.sigma * E - (p.gamma + p.delta) * I
    du[4] = p.gamma * I
    du[5] = p.delta * I
end

#========================================================
FUNCTION TO RUN THE MODEL
=========================================================#

# Define function to run the SEIRD model with a functional form for beta
# p must be of the form (beta0, delta, sigma, gamma, zeta)
# fixed_p must be of the form (sigma, gamma, zeta, population, E0, R0_recovered, D0)
# estimated_p must be of the form (prevalence, delta, R0_reproduction)
function run_seird_functional_form(fixed_p, estimated_p, train_length, smoothing_window = 7)

    # Retrieve fixed parameters
    sigma = fixed_p.sigma
    gamma = fixed_p.gamma
    zeta = fixed_p.zeta
    N = fixed_p.population
    E0 = fixed_p.E0
    R0_recovered = fixed_p.R0_recovered 
    D0 = fixed_p.D0

    # Retrieve estimated parameters
    prevalence = estimated_p.prevalence
    delta = estimated_p.delta
    R0_reproduction = estimated_p.R0_reproduction

    # Derive other parameters
    beta0 = R0_reproduction * (gamma + delta)
    I0 = max(1.0, prevalence * N)
    S0 = N - E0 - I0 - R0_recovered - D0

    # Define initial state
    init_state = [S0, E0, I0, R0_recovered, D0]
    # Define parameters for the ODE solver
    p = (beta0, delta, sigma, gamma, zeta)

    # Simulate extra days for the first 6 days to avoid errors when calculating the moving average
    extra_days = smoothing_window - 1
    tspan = (0, train_length + extra_days)

    # Define ODE problem 
    prob = ODEProblem(seird_functional!, init_state, tspan, p)
    # Solve ODE problem saving every day
    sol = solve(prob, Tsit5(), saveat=1.0, dense = false)
    
    return sol
end


# Convert csv to jld2 file
datafile = CSV.File(datadir("sims", "synthetic_mortality_ground_truth_exp.csv"))

# turn the data into a dataframe
df = DataFrame(datafile)

# Save the dataframe as a jld2 file
save(datadir("sims", "synthetic_mortality_ground_truth_exp.jld2"), "df", df)
