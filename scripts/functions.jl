#=============================================================
MODULE CONTAINING FUNCTIONS USED MULTIPLE TIMES IN THE PROJECT
==============================================================# 

module Functions

export loss_ude
export loss_mse

# Loss function using MSE
function loss_ude(p_all, predict_ude, data)
    pred = predict_ude(p_all)

    # Align lengths
    n = min(length(pred), length(data))
    pred = pred[1:n]
    data = data[1:n]

    # Mean squared error
    mse = sum((pred .- data).^2)/length(data)

    return mse, pred
end

# Loss function using MSE for evaluation of performance
function loss_mse(pred, data)

    # Mean squared error
    mse = sum((pred .- data).^2)/length(data)

    return mse
end

end