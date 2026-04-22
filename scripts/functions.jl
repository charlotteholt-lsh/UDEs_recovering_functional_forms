#=============================================================
MODULE CONTAINING FUNCTIONS USED MULTIPLE TIMES IN THE PROJECT
==============================================================# 

module Functions

export loss_ude
export loss_mse

# Loss function using NLL
function loss_ude(p_all, predict_ude, data)
    pred = predict_ude(p_all)

    # Align lengths
    n = min(length(pred), length(data))
    pred = pred[1:n]
    data = data[1:n]

    # Poisson negative log-likelihood
    ε = 1e-6
    nll = sum(pred .- data .* log.(pred .+ ε))

    # L2 penalty on NN weights (regularisation)
    # l2_penalty = 1e-4 * sum(abs2, p_all.nn_params)

    return nll, pred
end

# Loss function using MSE for evaluation of performance
function loss_mse(pred, data)

    # Mean squared error
    mse = sum((pred .- data).^2)/length(data)

    return mse
end

end