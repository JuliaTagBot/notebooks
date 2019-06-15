using Turing
using Distributions
using StatsPlots
using StatsBase
using Random

# Setup Turing
Turing.turnprogress(false)
Turing.setadbackend(:forward_diff)

# Binning Function
bin(data, edges) = fit(Histogram, data, edges);

# Sample Signal/Background
sample_signal(n; μ=1, σ=0) = rand(Normal(μ, σ), n)
sample_background(n; θ=1) = rand(Exponential(θ), n)

# Sample Generator
function sample_full(n; θ=1, μ=1, σ=0, r=1)
    return (sample_signal(round(Int, n * r); μ=μ, σ=σ),
            sample_background(n; θ=θ))
end

# Signal/Background Estimate Generator
function expected(n, binning; θ=1, μ=1, σ=0, r=1)
    s, b = sample_full(n; θ=θ, μ=μ, σ=σ, r=r)
    sₕ = bin(s, binning)
    bₕ = bin(b, binning)
    return sₕ, bₕ, merge(bₕ, sₕ)
end

# Fake Data Generator
function observed(α, n, binning; θ=1, μ=1, σ=0, r=1)
    s, b = sample_full(n; θ=θ, μ=μ, σ=σ, r=r)
    sₕ = bin(s, binning)
    sₕ.weights = map(w -> round(Int, α * w), sₕ.weights)
    return merge(sₕ, bin(b, binning))
end

# Fake Data Generator
function observed(α, sₕ, bₕ)
    dsₕ = deepcopy(sₕ)
    dsₕ.weights = map(w -> round(Int, α * w), dsₕ.weights)
    return merge(dsₕ, bₕ)
end

# True Signal Strength
true_α = 0.25

# Make Expected Counts and Fake Data
Random.seed!(1)
N = 4000
binning = 0:0.1:2.5
θ = 0.56
μ = 1.12
σ = 0.09
r = 0.1
signal, background, total = expected(N, binning; θ=θ, μ=μ, σ=σ, r=r)
data = observed(true_α, N, binning; θ=θ, μ=μ, σ=σ, r=r)

# "Bump Hunt" Plot
plot(total, title="Bump Hunt", label="expected signal")
plot!(background, label="expected background")
scatter!(data, label="data", color=:black)

# Counting Experiment Model
@model counting_experiment(data, signal, background) = begin
    # Signal Strength Prior
    α ~ Flat()

    # Draw a Poisson Process in each Bin
    for i in eachindex(data)
        data[i] ~ Poisson(α * signal[i] + background[i])
    end
end

iterations = 1000

# HMC:
ϵ = 0.001
τ = 1
hmc_sampler = HMC(iterations, ϵ, τ);

# MH:
mh_sampler = MH(iterations);

# NUTS:
adapts = 200
δ = 0.65
nuts_sampler = NUTS(iterations, adapts, δ)

# Sample Many Markov Chains
sample_many(n, args...) = mapreduce(_ -> sample(args...), chainscat, 1:n)

# Compute the Posterior
chains = 12
algorithm = mh_sampler
model_instance = counting_experiment(data.weights,
                                     signal.weights,
                                     background.weights)
observed_posterior = sample_many(chains, model_instance, algorithm)

println("true_α = $true_α")
println("median(α) = $(median(observed_posterior[:α].value))")
histogram(observed_posterior[:α], bins=70)

# 95% Confidence Limit for Signal Strength
limit(posterior; q=0.95) = quantile(posterior, q=[q])[1, 2]

# Observed Limit
observed_limit = limit(observed_posterior[:α])

chains = round(Int, chains / 3)
limit_samples = 100
expected_limits = Vector{Float64}(undef, limit_samples)
for i in 1:limit_samples
    sb = bin(sample_background(N; θ=θ), binning)
    toy_model = counting_experiment(sb.weights,
                                    signal.weights,
                                    background.weights)
    expected_limits[i] = limit(sample_many(chains, toy_model, algorithm))
end

histogram(expected_limits, bins=50)

expected_bands = quantile(expected_limits, [0.025, 0.16, 0.5, 0.84, 0.975])

plus_2σ = expected_bands[5]
excess_found = plus_2σ < observed_limit
println(excess_found ? "excess:" : "upper limit:")
println("  +2σ=", plus_2σ, " $(excess_found ? '<' : '>') obs=", observed_limit)

plus_5σ = quantile(expected_limits, 0.999999426697)

@model counting_experiment(data, control) = begin
    N = length(data)
  	signal = Vector(undef, N)
    background = Vector(undef, N)

 	  # Signal Strength Prior
    α ~ Flat()

    # Background in Control Region
    τ ~ Flat()

    # Fit Signal to Normal(μ, σ²)
    σ² ~ InverseGamma(2, 3)
  	μ ~ Normal(0, √σ²)

  	# Fit Background to Exponential(θ)
    θ ~ Flat()

    # Draw a Poisson Process in each Bin
    for i in 1:N
        # Fit Signal and Background
      	signal[i] ~ Normal(μ, σ²)
      	background[i] ~ Exponential(θ)
        # Estimate Expected Counts
        data[i] ~ Poisson(α * signal[i] + background[i])
    	  control[i] ~ Poisson(τ * background[i])
    end
end
