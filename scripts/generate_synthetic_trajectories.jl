
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
using DifferentialEquations
using ComponentArrays
using DSP
using Plots 

#========================================================
HELPER FUNCTIONS
=========================================================#

# Calculate a 7-day moving average
function moving_average(x)
    k = 7
    kernel = ones(k) / k
    if length(x) < k
        return zeros(length(x))
    end
    valid = DSP.conv(x, kernel)[k:end-k+1]
    # Returns a vector of length(x) where the first k-1 entries are zero
    return vcat(zeros(k-1), valid)
end

# Build the file name for simulation
function file_name(fixed_p, estimated_p)
    params = (
        pop  = Int(round(fixed_p.population)),
        E0   = round(fixed_p.E0, sigdigits=5),
        R0   = round(fixed_p.R0_recovered, sigdigits=5),
        D0   = round(fixed_p.D0, sigdigits=5),
        sig  = round(fixed_p.sigma, sigdigits=5),
        gam  = round(fixed_p.gamma, sigdigits=5),
        zet  = round(fixed_p.zeta, sigdigits=5),
        prev = round(estimated_p.prevalence, sigdigits=5),
        del  = round(estimated_p.delta, sigdigits=5),
        R0r  = round(estimated_p.R0_reproduction, sigdigits=5),
    )
    return "synthetic_" * savename(params; connector="_", sort=false)
end

function convert_to_daily_and_smooth(traj, smoothing_window)
    # Convert to daily values by taking the difference between consecutive entries
    daily_values = [0.0; diff(traj)]
    # Smooth the daily values using a moving average
    smoothed_daily_values = moving_average(daily_values)
    # Remove the first few entries that are zero due to the moving average
    smoothed_daily_values = smoothed_daily_values[smoothing_window:end]
    return smoothed_daily_values
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
function run_seird_functional_form(fixed_p, estimated_p, obs_length, smoothing_window = 7)

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
    p = ComponentArray(
        beta0 = beta0,
        delta = delta,
        sigma = sigma,
        gamma = gamma,
        zeta = zeta
    )

    # ONLY NEED THIS STEP IF I AM DOING A MOVING AVERAGE
    # Simulate extra days for the first 6 days to avoid errors when calculating the moving average
    #extra_days = smoothing_window - 1
    # Remove a day so when we take the difference we have the obs_length number of data points
    #tspan = [0, obs_length + extra_days -1]

    tspan = [0, obs_length - 1]

    # Define ODE problem 
    prob = ODEProblem(seird_functional!, init_state, tspan, p)
    # Solve ODE problem saving every day
    sol = solve(prob, Tsit5(), saveat=1.0, dense = false)
    
    return sol
end

function generate_synthetic_data(fixed_p, estimated_p, obs_length, smoothing_window = 7)
    # Run model
    sim = run_seird_functional_form(fixed_p, estimated_p, obs_length, smoothing_window)

    # Extract raw states 
    s_traj = sim[1, :]
    e_traj = sim[2, :]
    i_traj = sim[3, :]
    r_traj = sim[4, :]
    d_traj = sim[5, :]



    # Save the result
	fname = file_name(fixed_p, estimated_p) * ".jld2"

	mkpath(datadir("synthesised_trajectories"))


	save(datadir("synthesised_trajectories", fname),
		"fixed_p", fixed_p, "estimated_p", estimated_p, "days", 1:obs_length, 
        "susceptible", s_traj, "exposed", e_traj, "infectious", i_traj, "recovered", r_traj, "deaths", d_traj)

    println("Finished generating synthetic data for $(fname)")

    return s_traj, e_traj, i_traj, r_traj, d_traj

end


#========================================================
DEFINE PARAMETERS AND GENERATE SYNTHETIC DATA
=========================================================#

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


# EXTRACTED USING GROUND_TRUTH_VALUES FOR MA IN THE PYTHON CODE
prevalence = exp(-11.468967)
R0_reproduction = 5.284404
const delta = exp(-8.941224)

fixed_p = ComponentArray(sigma = sigma, 
                        gamma = gamma, 
                        zeta = 0.02, 
                        population = N,
                        E0 = E0, 
                        R0_recovered = R0, 
                        D0 = D0)
estimated_p = ComponentArray(prevalence = prevalence, 
                            delta = delta, 
                            R0_reproduction = R0_reproduction)

generate_synthetic_data(fixed_p, estimated_p, 365)


# Convert csv to jld2 file
datafile = CSV.File(datadir("sims", "synthetic_mortality_ground_truth_exp.csv"))

# turn the data into a dataframe
df = DataFrame(datafile)

# Save the dataframe as a jld2 file
save(datadir("sims", "synthetic_mortality_ground_truth_exp.jld2"), "df", df)


# Extract infectious trajectoryfrom JLD2 file
dataset = load(datadir("synthesised_trajectories", "synthetic_pop=6892503_E0=0.0_R0=0.0_D0=0.0_sig=0.333_gam=0.1_zet=0.02_prev=1.04e-5_del=0.000131_R0r=5.28.jld2"))


# Plot infections
infections = dataset["infectious"]
display(plot(1:length(infections), infections, label="Generated data", xlabel="Days", ylabel="Infections", title="Infections over time"))
