#---Hyperparameter:
mutable struct Hyperparameters
    zdim::Int
    n_restarts::Int
    epochs::Int
    batchsize::Int
    η::Union{AbstractFloat, Int}
    λ::Union{AbstractFloat, Int}
    ϵ::Union{AbstractFloat, Int}
    M::Int

    # Constructor with default values
    function Hyperparameters(; zdim::Int=10, n_restarts::Int=1, epochs::Int=50, batchsize::Int=2^9, η::Union{AbstractFloat, Int}=0.01, λ::Union{AbstractFloat, Int}=0.1, ϵ::Union{AbstractFloat, Int}=0.001s, M::Int=1)
        new(zdim, n_restarts, epochs, batchsize, η, λ, ϵ, M)
    end 
end

#---MetaData:
mutable struct MetaData
    obs_df::DataFrame
    featurename::Union{Nothing, Vector{String}}
    Top_features::Union{Nothing, Dict{String, DataFrame}}

    # Constructor with default values
    function MetaData(; obs_df::DataFrame=DataFrame(), featurename::Union{Nothing, Vector{String}}=nothing, Top_features::Union{Nothing, Dict{String, DataFrame}}=nothing)
        new(obs_df, featurename, Top_features)
    end 
end

#---BAE architecture:
mutable struct BoostingAutoencoder
    coeffs::AbstractMatrix{Float32}
    decoder::Union{Chain, Dense}
    HP::Hyperparameters
    Z::Union{Nothing, Matrix{Float32}}
    Z_cluster::Union{Nothing, Matrix{Float32}}
    UMAP::Union{Nothing, Matrix{Float32}}

    # Constructor to allow initializing Z as nothing
    function BoostingAutoencoder(; coeffs::Matrix{Float32}, decoder::Union{Chain, Dense}, HP::Hyperparameters)
        new(coeffs, decoder, HP, nothing, nothing, nothing)
    end
end
(BAE::BoostingAutoencoder)(X) = BAE.decoder(transpose(BAE.coeffs) * X)
Flux.@functor BoostingAutoencoder

function Base.summary(BAE::BoostingAutoencoder)
    HP = BAE.HP
    println("Initial hyperparameter for constructing and training a BAE:
     latent dimensions: $(HP.zdim),
     number of encoder re-starts: $(HP.n_restarts),
     training epochs: $(HP.epochs),
     batchsize: $(HP.batchsize),
     learning rate for decoder parameter: $(HP.η),
     weight decay parameter for decoder parameters: $(HP.λ),
     step size for boosting updates: $(HP.ϵ),
     number of boosting iterations: $(HP.M)."
    )
end

function generate_BAEdecoder(p::Int, HP::Hyperparameters; soft_clustering::Bool=true, modelseed::Int=42)
    Random.seed!(modelseed)

    if soft_clustering
        decoder = Chain(x -> softmax(split_vectors(x)), Dense(2*HP.zdim => p, tanh_fast), Dense(p => p));
        @info "Decoder with 4 layers: First two layers (split_softmax) are non trainable, third layer is Dense(2*HP.zdim => p, tanh), fourth layer is Dense(p => p)"
    else 
        decoder = Chain(Dense(HP.zdim => p, tanh), Dense(p => p));
        @info "Decoder with 2 trainable layers: First layer is Dense(HP.zdim => p, tanh), second layer is Dense(p => p)"
    end
    return decoder
end