# Note: unbiased==false is what R uses.
function acvf(x::Array, max_lag::Int, unbiased::Bool)
    x_demeaned = x - mean(x)
    a = zeros(max_lag + 1)
    n = length(x_demeaned)
    for lag = 0:max_lag
        for t = 1:n-lag
            a[lag + 1] += x_demeaned[t + lag] * x_demeaned[t]
        end
        if unbiased
            a[lag + 1] /= (n - lag)
        else
            a[lag + 1] /= n
        end
    end
    return a
end
acvf(x::Array, max_lag::Int) = acvf(x, max_lag, true)
acvf(x::Array) = acvf(x, 40, true)


function acf(x::Array, max_lag::Int, unbiased::Bool)
    a = acvf(x, max_lag)
    return a / a[1]
end
acf(x::Array, max_lag::Int) = acf(x, max_lag, true)
acf(x::Array) = acf(x, 40, true)

type ARIMAModel
    ar::Array
    d::Int
    ma::Array
    sigma::Float
end

# Should have checks for stationary ARIMA models
function arima_sim(n, ar, d, ma, sigma, burn_in)
    x = zeros(n + burn_in)
    w = randn(n + burn_in) * sigma
    p = length(ar)
    q = length(ma)
    s = max(p, q)
    for t = s:n+burn_in-1
        x[t + 1] = dot(ar, x[t-p+1:t]) + dot(ma, w[t-q+1:t]) + w[t]
    end
    for i in 1:d
        x = cumsum(x)
    end
    return x[end-n+1:end]
end
arima_sim(n, mod::ARIMAModel) = arima_sim(n, mod.ar, mod.d, mod.ma, mod.sigma, 500)

type StateSpaceModel
    trans::Array
    trans_err::Array
    obs::Array
    obs_err::Array
    state0::Array
    error0::Array
end

type KalmanFilter
   predicted::Array
   filtered::Array
   like::Float 
end

function kalman(x::Array, model::StateSpaceModel)
    # transpose data to make equations a little clearer
    x = x'
    N = size(x)[2]
    Trans, Trans_err = model.trans, model.trans_err
    Obs, Obs_err = model.obs, model.obs_err
    state0, Pred_err = model.state0, model.error0
    state_pred = zeros(size(state0)[1], N)
    state = zeros(size(state_pred))
    smooth = zeros(size(state_pred))
    # Kalman prediction and smoothing uses a loop-and-a-half
    # first prediction
    state_pred[:, 1] = Trans * state0
    Pred_err_0 = Trans * Pred_err * Trans' + Trans_err
    # first update
    KGain = Pred_err_0 * Obs' * inv(Obs * Pred_err_0 * Obs' + Obs_err)
    state[:, 1] = state_pred[:, 1] + KGain * (x[:, 1] - Obs * state_pred[:, 1])
    Pred_err = (eye(size(KGain)[1]) - KGain * Obs) * Pred_err_0
    # filter the rest of the series
    for t=2:N
        # predict
        state_pred[:, t] = Trans * state[:, t-1]
        Pred_err_0 = Trans * Pred_err * Trans' + Trans_err
        # update
        KGain = Pred_err_0 * Obs' * inv(Obs * Pred_err_0 * Obs' + Obs_err)
        state[:, t] = state_pred[:, t] + KGain * (x[:, t] - Obs * state_pred[:, t])
        Pred_err = (eye(size(KGain)[1]) - KGain * Obs) * Pred_err_0
    end
    # smoothing
    smooth[:, N] = state[:, N]
    
    return state_pred, state
end


x = arima_sim(100, [0.3, 0.1], 0, [0], 1.3, 500)
phi1 = 0.35
phi2 = 0.08
v = 1.2
mod_ar = StateSpaceModel(
            [phi1 phi2; 1 0],   # trans
            [v 0; 0 0],         # trans_err
            [1 0],              # obs
            [0],                # obs_err
            [0, 0],             # state0
            [1e6 1e6; 1e6 1e6]) # error0
res = kalman(x, mod_ar)

        
        