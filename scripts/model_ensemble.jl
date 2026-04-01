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
    dataset = load(datadir("sims", "synthetic_mortality_ground_truth_exp.jld2"))
    # Just use data with strongest behavioural response (zeta = 0.02)
    df = dataset["df"]
    obs = df[!, "y_zeta_0.02"]
    days = df[!, "days"]

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
            # Extract the predicted mortalities for the training data
            D_pred = pred[5, 1:length(obs)]
            daily_deaths_pred = [0.0; diff(D_pred)]
            push!(all_predictions, daily_deaths_pred)
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
    label="Data", xlabel="Day", ylabel="Daily deaths", title="$plot_title")
    annotate!(pl, x[end], maximum(obs), text("MSE: $(round(mse, digits=4))", 9, :right))
    plot!(pl, x, median_prediction, color=:red, linewidth=2, ribbon = ((median_prediction - lower_quantile), (upper_quantile - median_prediction)), label="Median prediction")
    display(pl)

    # Save the plot
    savefig(pl, datadir("sims", "ude", sim_name, "prediction_plot.png"))

    return pl

end


sim_name = "synthesised_MA_input_death_time_hidden_dims_5_RB_solve_no_param_check"
plot_title = "Optimal prediction 250326"
plot_simulation(sim_name, plot_title)






