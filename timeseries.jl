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