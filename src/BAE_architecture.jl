#---Hyperparameter:
mutable struct Hyperparameters
    zdim::Int
    n_runs::Int
    max_iter::Int
    tol::Union{AbstractFloat, Nothing}
    batchsize::Int
    η::Union{AbstractFloat, Int}
    λ::Union{AbstractFloat, Int}
    ϵ::Union{AbstractFloat, Int}
    M::Int

    # Constructor with default values
    function Hyperparameters(; zdim::Int=10, n_runs::Int=2, max_iter::Int=1000, tol::Union{AbstractFloat, Nothing}=1e-5, batchsize::Int=2^9, η::Union{AbstractFloat, Int}=0.01, λ::Union{AbstractFloat, Int}=0.1, ϵ::Union{AbstractFloat, Int}=0.001, M::Int=1)
        new(zdim, n_runs, max_iter, tol, batchsize, η, λ, ϵ, M)
    end 
end

#---MetaData:
"""
    mutable struct MetaData

A mutable struct that encapsulates metadata associated with observations and features in a dataset.

# Fields

- `obs_df::DataFrame`: A DataFrame containing additional information about the observations, e.g., cell types of cells or sender and receiver cell types in case observations are cell pairs. Default is an empty `DataFrame`.
- `featurename::Union{Nothing, Vector{String}}`: An optional vector of strings representing the names of the features, e.g. gene names or ligand-receptor interaction names. Default is `nothing`.
- `Top_features::Union{Nothing, Dict{String, DataFrame}}`: An optional dictionary where training results of a BAE about the selected features can be stored. The keys are numbers (strings) representing the different clusters of a BAE model and the values are DataFrames consist of the importance scores of features per cluster, the actual feature scores and the feature names. Default is `nothing`.

# Constructor

- `MetaData(; obs_df::DataFrame=DataFrame(), featurename::Union{Nothing, Vector{String}}=nothing, Top_features::Union{Nothing, Dict{String, DataFrame}}=nothing)`: 
  Creates a new instance of `MetaData` with default or user-defined values for each field.

# Example Usage

```julia
# Creating an empty MetaData instance
MD = MetaData()

# Creating a MetaData instance with specific observations and feature names
MD = MetaData(obs_df=obs_df, featurename=["feature1", "feature2"])

# Creating a MetaData instance with all fields defined
MD = MetaData(obs_df=obs_df, featurename=["feature1", "feature2"], Top_features=Dict("1" => top_feature_df1))
"""
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
     number of encoder re-starts: $(HP.n_runs),
     maximum number of training epochs per run: $(HP.max_iter),
     tolerance: $(HP.tol),
     batchsize: $(HP.batchsize),
     learning rate for decoder parameter: $(HP.η),
     weight decay parameter for decoder parameters: $(HP.λ),
     step size for boosting updates: $(HP.ϵ),
     number of boosting iterations: $(HP.M)."
    )
end

"""
    generate_BAEdecoder(p::Int, HP::Hyperparameters; soft_clustering::Bool=true, modelseed::Int=42)

Generates a decoder for a Boosting Autoencoder (BAE) based on the specified hyperparameters and configuration options.

# Arguments

- `p::Int`: The output dimension of the decoder, typically corresponding to the number of features in the data.
- `HP::Hyperparameters`: An instance of the `Hyperparameters` struct containing settings such as the latent dimension `zdim` and other training parameters (for more details see the documentation of `Hyperparameters`).

# Keyword Arguments

- `soft_clustering::Bool=true`: If `true`, the decoder will include a soft clustering component using a `softmax` function applied to the output of the split operation. This results in a decoder with three layers, where the first one is non-trainable and involve `softmax` and vector splitting. If `false`, the decoder will consist of only two trainable layers.
- `modelseed::Int=42`: A seed for the random number generator, ensuring reproducibility of the decoder's initialization.

# Returns

- `decoder`: A Flux `Chain` object representing the decoder network. Depending on the `soft_clustering` argument:
  - If `soft_clustering=true`, the decoder consists of three layers:
    1. A non-trainable `softmax` layer applied to split vectors.
    2. A Dense layer with `2*HP.zdim` input units and `p` output units with `tanh` activation.
    3. A Dense layer with `p` input units and `p` output units.
  - If `soft_clustering=false`, the decoder consists of two trainable layers:
    1. A Dense layer with `HP.zdim` input units and `p` output units with `tanh` activation.
    2. A Dense layer with `p` input units and `p` output units.

# Example Usage

```julia
HP = Hyperparameters(zdim=5, epochs=100, η=0.001)

# Generate a decoder with soft clustering
decoder_with_soft_clustering = generate_BAEdecoder(10, HP, soft_clustering=true)

# Generate a decoder without soft clustering
decoder_without_soft_clustering = generate_BAEdecoder(10, HP, soft_clustering=false)
"""
function generate_BAEdecoder(p::Int, HP::Hyperparameters; soft_clustering::Bool=true, modelseed::Int=42)
    Random.seed!(modelseed)

    if soft_clustering
        decoder = Chain(x -> softmax(split_vectors(x)), Dense(2*HP.zdim => p, tanh_fast), Dense(p => p));
        @info "Decoder with 3 layers: First two layers (split_softmax) are non trainable, third layer is Dense(2*HP.zdim => p, tanh), fourth layer is Dense(p => p)"
    else 
        decoder = Chain(Dense(HP.zdim => p, tanh), Dense(p => p));
        @info "Decoder with 2 trainable layers: First layer is Dense(HP.zdim => p, tanh), second layer is Dense(p => p)"
    end
    return decoder
end