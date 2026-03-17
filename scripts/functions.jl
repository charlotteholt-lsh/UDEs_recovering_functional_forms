module Functions

export loss_ude
export loss_mse

# Loss function using NLL with L2 regularisation.
function loss_ude(p_all, _)
    pred = predict_ude(p_all)

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