#=============================================================
SCRIPT TO ANALYSE THE PERFORMANCE OF THE UDE MODEL IN TRAINING
=============================================================#
using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")
include(joinpath(@__DIR__, "functions.jl"))
using .Functions
using JLD2
using Plots
using Statistics
using StatsBase
using Distributions
using ComponentArrays
using DataFrames

#=============================================================
FUNCTION TO GENERATE PLOTS OF SIMULATIONS
=============================================================#

function plot_simulation(sim_name, plot_title)

    # Load the observed data
    dataset = load(datadir("synthesised_trajectories", "synthetic_pop=6892503_E0=0.0_R0=0.0_D0=0.0_sig=0.333_gam=0.1_zet=0.02_prev=1.04e-5_del=0.000131_R0r=5.28.jld2"))
    # Just use infectious trajectory
    obs = dataset["infectious"]
    days = dataset["days"]

    # Define the root file path
    root = datadir("sims", "ude", sim_name)
    # Create empty array to store predictions with initial mortalities 0
    all_predictions = []
    # Read all files/folders in the root directory
    for filename in readdir(root)
        # Only include directories
        if isdir(joinpath(root, filename))
            # Extract results, predictions and losses
            results = load(datadir("sims", "ude", sim_name, filename, "results.jld2"))
            pred = results["prediction"]
            # Extract the predicted infectious trajectory for the training data
            i_traj = pred[3, 1:length(obs)]
            push!(all_predictions, i_traj)
        end
    end

    # create a prediction matrix that stacks each simulation on top of each other
    # e.g. 100 rows (one for each simulation) and 123 columns (one for each day of the training data)
    prediction_matrix = hcat(all_predictions...)'

    # Find the median prediction and IQR across all simulations
    # Find the mdeian aggregating over the rows (i.e. across all simulations) for each column (i.e. for each day of the training data)
    median_prediction = median(prediction_matrix, dims=1)
    # Convert to a vector
    median_prediction = vec(median_prediction)

    # Find the upper and lower quantiles across all simulations for each day of the training data
    # Compute per-column explicitly - (creating views rather than copies)
    upper_quantile = [quantile(view(prediction_matrix, :, j), 0.75) for j in axes(prediction_matrix, 2)]
    lower_quantile = [quantile(view(prediction_matrix, :, j), 0.25) for j in axes(prediction_matrix, 2)]

    # Evaluate MSE using the median
    mse = Functions.loss_mse(median_prediction, obs)

    # Save to a JLD2 file
    save(datadir("sims", "ude", sim_name, "summary.jld2"), "median_prediction", median_prediction, "upper_quantile", upper_quantile, "lower_quantile", lower_quantile, "mse", mse)

    # Create plot
    x = days[1:length(prediction_matrix[1, :])]
    pl = scatter(x, obs[1:length(prediction_matrix[1, :])], color=:black, markersize=2,
    label="Data", xlabel="Day", ylabel="Daily deaths", title="$plot_title", legend=:topright)
    x_annot = x[end] - 0.02 * (x[end] - x[1])
    y_annot = maximum(obs[1:length(prediction_matrix[1, :])]) * 0.88
    annotate!(pl, x_annot, y_annot, text("MSE: $(round(mse, digits=4))", 9, :right))
    plot!(pl, x, median_prediction, color=:red, linewidth=2, ribbon = ((median_prediction - lower_quantile), (upper_quantile - median_prediction)), label="Median prediction")
    display(pl)

    # Save the plot
    savefig(pl, datadir("sims", "ude", sim_name, "prediction_plot.png"))

    return pl

end

#========================================================
PRODUCE ENSEMBLE PLOTS
=========================================================# 
# sim_name = "synthesised_use_normalised_infections_optimal_250326"
# plot_title = "Optimal prediction 5 inputs 250326"
# plot_simulation(sim_name, plot_title)

#=============================================================
FUNCTION TO PLOT APPROXIMATED FUNCTION AGAINST SYNTHESISED DATA
=============================================================#

function plot_individual_traj(sim_num, sim_name, synthesised_data)

    # Load the observed data and varying parameters
    dataset = load(datadir("synthesised_trajectories", synthesised_data))
    varying_p = ComponentArray(
        population = dataset["varying_p"]["population"],
        prevalence = dataset["varying_p"]["prevalence"],
        delta = dataset["varying_p"]["delta"],
        R0_reproduction = dataset["varying_p"]["R0_reproduction"],
        zeta = dataset["varying_p"]["zeta"]
    )

    # Derive beta0 specific to current trajectory
    beta0 = varying_p.R0_reproduction * (gamma + varying_p.delta)

    # Just use infectious trajectory
    obs = dataset["infectious"]
    days = dataset["days"]

    # Define the root file path
    root = datadir("sims", "ude_multiple", sim_name, sim_num, synthesised_data, "results.jld2")
    plot_dir = dirname(root)

    # Extract predictions for epidemic trajectory
    results = load(root)
    pred = results["infectious_traj_prediction"]

    # Extract the predicted infectious trajectory for the training data
    i_traj = pred[3, 1:length(obs)]

    # Extract beta trajectory
    beta_pred = results["beta_traj"]

    # Define beta function
    function true_beta(I)
        arg_exp = clamp(varying_p.zeta * varying_p.delta * I, -50.0, 50.0)
        beta = beta0 * exp(-arg_exp)
        return beta
    end

    # Generate true beta values
    beta_true = true_beta.(i_traj)
    
    # Ensure both are vectors
    if ndims(beta_true) > 1
        beta_true = vec(beta_true)
    end
    if ndims(beta_pred) > 1
        beta_pred = vec(beta_pred)
    end
    
    # Create trajectory plot
    traj_plot = plot(days[1:length(i_traj)], obs[1:length(i_traj)], color=:black, markersize=2, label="Data", 
    xlabel="Day", ylabel="Infectious individuals", title="Infectious trajectory for $(sim_name)", legend=:topright)
    plot!(traj_plot, days[1:length(i_traj)], i_traj, color=:red, linewidth=2, label="Predicted trajectory")
    display(traj_plot)

    # Save the plot
    savefig(traj_plot, joinpath(plot_dir, "traj_plot.png"))

    # Create beta plot
    beta_plot = plot(days[1:length(beta_true)], beta_true, color=:blue, linewidth=2, label="True beta", 
    xlabel="Day", ylabel="Beta", title="Beta trajectory for $(sim_name)", legend=:topright)
    plot!(beta_plot, days[1:length(beta_pred)], beta_pred, color=:red, linewidth=2, label="Predicted beta")
    display(beta_plot)

    # Save the plot
    savefig(beta_plot, joinpath(plot_dir, "beta_plot.png"))

    return traj_plot, beta_plot
end

#========================================================
PRODUCE PLOTS OF SIMULATIONS
=========================================================# 


sim_num = "simulation_v1"
sim_name = "synthesised_use_5_inputs_optimal_250326"
for filename in readdir(datadir("sims", "ude_multiple", sim_name, sim_num), endswith=".jld2")
    if isdir(joinpath(datadir("sims", "ude_multiple", sim_name, sim_num, filename)))
        plot_individual_traj(sim_num, sim_name, filename)
    end
end

