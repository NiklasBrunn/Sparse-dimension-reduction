#---Componentwise boosting functions:
"""
    get_unibeta(X::AbstractMatrix{T}, y::AbstractVector{T}, denom::AbstractVector{T}, n::Int, p::Int) where T

Compute the univariate regression coefficients for each feature/predictor.

# Arguments
- `X::AbstractMatrix{T}`: A matrix of predictors with dimensions `n` by `p`, where `n` is the number of observations (rows) and `p` is the number offeatures/predictors (columns).
- `y::AbstractVector{T}`: A vector of responses with length `n`.
- `denom::AbstractVector{T}`: A vector of denominators with length `p`, typically corresponding to `n - 1` if the data is already feature standardized, or another value specific to each predictor (usually the feature/predicor variances).
- `n::Int`: The number of observations (rows in `X` and elements in `y`).
- `p::Int`: The number of features/predictors (columns in `X` and elements in `denom`).

# Returns
- `unibeta::Vector{T}`: A vector of length `p` containing the univariate regression coefficients for each predictor.

# Description
This function calculates the univariate regression coefficients for each predictor in `X` using the corresponding elements of the response vector `y`. For each predictor `j`, the coefficient is computed as the sum of the products of the predictor values and the response values across all observations, divided by the corresponding element in `denom`. The coefficient corresponds to the univariate ordinary linear least squares estimators.

The univariate regression coefficient for predictor `j` is given by:

    unibeta[j] = (sum(X[i, j] * y[i] for i in 1:n)) / denom[j]
"""
function get_unibeta(X::AbstractMatrix{T}, y::AbstractVector{T}, denom::AbstractVector{T}, n::Int, p::Int) where T

    unibeta = zeros(eltype(X), p)

    for j = 1:p
        for i = 1:n
            unibeta[j] += X[i, j] * y[i]
        end

        unibeta[j] /= denom[j] #n - 1
    end

    return unibeta
end

"""
    compL2Boost!(BAE::BoostingAutoencoder, l::Int, X::AbstractMatrix{T}, y::AbstractVector{T}, denom::AbstractVector{T}) where T

Perform componentwise L2-boosting with linear base learners given the coefficients `BAE:coeffs` of a Boosting Autoencoder model, the data matrix `X` with observations as rows and features as columns and the current responses `y`.

# Arguments
- `BAE::BoostingAutoencoder`: A boosting autoencoder object containing model coefficients and hyperparameters (for more details see the documentation of `BoostingAutoencoder` and `Hyperparameters`).
- `l::Int`: The index of the latent dimension vetor for the BAE to update.
- `X::AbstractMatrix{T}`: A matrix of features/predictors with dimensions `n` by `p`, where `n` is the number of observations and `p` is the number of features/predictors.
- `y::AbstractVector{T}`: A vector of responses with length `n`.
- `denom::AbstractVector{T}`: A vector of denominators with length `p`, typically corresponding to `n - 1` or another value specific to each predictor.

# Description
This function implements an iterative componentwise L2-boosting procedure for updating the coefficients of a BAE model. The procedure works as follows:

1. Compute the residual vector as the difference between the responses `y` and the current fit (based on the predictor matrix `X` and current coefficients `β` for component `l` corresponding to one row of the encoder weight matrix of the BAE).
2. Calculate the univariate ordinary least squares estimators (`unibeta`) for each predictor to fit the residuals.
3. Select the predictor that optimizes the current fit based on the maximum squared univariate estimator weighted by the corresponding `denom` value. This is equivalent to choosing the predictor that minimizes the residual sum of squares when adding the different features/predictors to the model.
4. Update the coefficients `β` by adding a rescaled version of the selected OLS estimator (`unibeta`) using the boosting step size (`ϵ`), and include the selected predictor index in the set of non-zero coefficients. Note that all nonzero coefficients receive updates in each iteration.
5. Repeat the above steps for a number of boosting iterations as specified by `BAE.HP.M`.

# Side Effects
- Updates the coefficients `β` of the BAE component `l` in-place within the `BAE` object.

# Notes
- This function is intended to be used within the context of training a BAE.
"""
function compL2Boost!(BAE::BoostingAutoencoder, l::Int, X::AbstractMatrix{T}, y::AbstractVector{T}, denom::AbstractVector{T}) where T
    n, p = size(X)

    β = BAE.coeffs[:, l]

    update_inds = findall(x->x!=0, β)

    for step in 1:BAE.HP.M
        res = y .- (X * β)

        unibeta = get_unibeta(X, res, denom, n, p) # Currently the faster option
        #unibeta = [dot(X[:, j], res) / denom[j] for j in 1:p] 

        optindex = findmax(unibeta.^2 .* denom)[2]

        update_inds = union(update_inds, optindex)
        BAE.coeffs[update_inds, l] .+= unibeta[update_inds] .* BAE.HP.ϵ  

    end

end

"""
    disentangled_compL2Boost!(BAE::BoostingAutoencoder, batch::AbstractMatrix{T}, grads::AbstractMatrix{T}) where T

Perform disentangled componentwise L2-boosting updates of the encoder weights of a Boosting Autoencoder (BAE).

# Arguments
- `BAE::BoostingAutoencoder`: A boosting autoencoder object containing model coefficients and hyperparameters (for more details see the documentation of `BoostingAutoencoder` and `Hyperparameters`).
- `batch::AbstractMatrix{T}`: A matrix representing a batch of input data with dimensions `n` by `p`, where `n` is the number of observations and `p` is the number of features.
- `grads::AbstractMatrix{T}`: Gradient of the scalar valued BAE reconstruction loss function w.r.t. the latent representation of the data in matrix form. The dimensions corresponding to the BAE's latent space. The negative gradients are used to determine the pseudo-responses for the boosting updates.

# Description
This function implements a disentangled componentwise L2-boosting procedure on a BAE, aiming to update the encoder weights/coefficients associated with each latent dimension. The procedure is designed to maintain disentanglement between different latent dimensions during the training process.

The procedure works as follows:

1. Compute the denominators (`denom`) as the sum of squared values in each column of the batch matrix (proportional to the variances of the features).
2. Compute a scaled version of the negative gradients (`Y`) for each latent dimension.
3. For each latent dimension `l` (from 1 to `BAE.HP.zdim`):
    - Identify the indices of latent dimensions with zero coefficients, plus the current dimension `l` and store them in a vector.
    - If all latent dimensions are present in the vector, use the standardized negative gradient (`l`-th row of `Y`) as the pseudo-responses (`y`).
    - If not all latent dimensions are present in the vector, compute the optimal ordinary least squares fit of the current latent representation (zerocolumns are excluded) to the scaled negative gradient using Cholesky decomposition. Then, compute the pseudo-responses `y` as the difference between the standardized negative gradient (`l`-th row of `Y`) and the fit.
    - Apply componentwise L2-boosting (`compL2Boost!`) to update the coefficients corresponding to the current latent dimension `l` using the computed pseudo-responses.

# Side Effects
- Updates the coefficients of the BAE in-place, modifying the `BAE.coeffs` matrix.

# Notes
- This function is intended for use in scenarios where a BAE is trained to maintaining disentangled latent representations, i.e. for structuring the latent space.
- The function employs a regularization term (using a small identity matrix) during the Cholesky decomposition to prevent numerical issues, such as singular matrices.
"""
function disentangled_compL2Boost!(BAE::BoostingAutoencoder, batch::AbstractMatrix{T}, grads::AbstractMatrix{T}) where T

    denom = vec(sum(batch .^ 2, dims=1))

    Y = scale(-grads, dims=2)

    for l in 1:BAE.HP.zdim

        Inds = union(find_zero_columns(BAE.coeffs), l)

        if length(Inds) == BAE.HP.zdim
            y = Y[l, :]

            compL2Boost!(BAE, l, batch, y, denom)

        else
            curdata = batch * BAE.coeffs[:, Not(Inds)]
            curtarget = Y[l, :]
            
            m = size(curdata, 2)
            Λ = convert(eltype(batch), 1.0e-6) .* Matrix(I, m, m) # Regularization term that prevents singular matrices (rarely happens)
            XtX = curdata' * curdata .+ Λ
            XtY = curdata' * curtarget
            
            curestimate = ldiv!(cholesky(XtX, Val(true)), XtY)

            y = scale(curtarget .- (curdata * curestimate))

            compL2Boost!(BAE, l, batch, y, denom)
        end
        
    end

end

#---training function for a BAE with the disentanglement constraint:
"""
    train_BAE!(X::AbstractMatrix{T}, BAE::BoostingAutoencoder; MD::Union{Nothing, MetaData}=nothing, track_coeffs::Bool=false, save_data::Bool=false, data_path::Union{Nothing, String}=nothing, batchseed::Int=42) where T

Train a Boosting Autoencoder (BAE) model using the given input data matrix `X` and hyperparameters stored as part of the BAE structure. 
The hyperparameters for training can be adapted in the definition of the `BAE.HP` field  of the BAE structure. 
The hyperparameters for training a BAE model are: The number of latent dimensions `BAE.HP.zdim`, the number of restarts `BAE.HP.n_restarts`,  the number of training epochs `BAE.HP.epochs`, 
the batch size of the mini-batches used for computing the parameter updates `BAE.HP.batchsize`, the learning rate for the decoder optimizer `BAE.HP.η`, the regularization parameter for decoder updates `BAE.HP.λ`, 
the step width for the boosting component `BAE.HP.ϵ`, the number of boosting steps per training iteration `BAE.HP.M`.
The training process is performed for `BAE.HP.epochs` epochs, with a batch size of `BAE.HP.batchsize`. 
If `BAE.HP.n_restarts` > 1, the training process is repeated `BAE.HP.n_restarts` times, each time starting with a zero initialization of the encoder weights, training for `BAE.HP.epochs` epochs. 
The training process is optimized using the AdamW optimizer. 
The training process is tracked by the mean batch loss per epoch, the entanglement score per epoch, the sparsity level per epoch, and the clustering score (silhouette score of the clustering) per epoch. 
Optionally, the coefficient updates can be tracked during training iterations. 
The training data can be saved at each epoch.

# Arguments
- `X::AbstractMatrix{T}`: The input data matrix of type `T`.Rows correspond to the observational units (e.g. cell pairs or cells) and columns to the features (e.g. interactions or genes). For optimizing training speed, the input data matrix should be of type `Float32`.
- `BAE::BoostingAutoencoder`: The BoostingAutoencoder model to train, including hyperparameters.
- `MD::Union{Nothing, MetaData}=nothing`: Optional metadata for the training process, including the featurenames.
- `track_coeffs::Bool=false`: Whether to track the coefficient updates during training iterations.
- `save_data::Bool=false`: Whether to save data. The data that is saved during training consists of the encoder weights, the mean batch loss per epoch, the entanglement score per epoch, the sparsity level per epoch and the clustering score per epoch.  If `true`, the data will be saved in the provided `data_path` at each epoch. In addition, selected features per cluster are stored as .csv files. This increases the memory usage!
- `data_path::Union{Nothing, String}=nothing`: The path to save the data.
- `batchseed::Int=42`: A seed for random number generation.

# Returns
- `output_dict::Dict{String, Union{Vector{eltype(X)}, Vector{Any}}}`: A dictionary containing the training results.
"""
function train_BAE!(X::AbstractMatrix{T}, BAE::BoostingAutoencoder; MD::Union{Nothing, MetaData}=nothing, track_coeffs::Bool=false, save_data::Bool=false, data_path::Union{Nothing, String}=nothing, batchseed::Int=42) where T

    Random.seed!(batchseed)

    object_type = eltype(X)

    if object_type != Float32
        @warn "The input data matrix is not of type Float32. This might lead to a slower training process."
    end

    if save_data
        if isnothing(data_path) || !isdir(data_path)
            @error "Please provide a valid path to save the training data."
        else
            if !isdir(data_path * "BAE_results_data")
                mkdir(data_path * "BAE_results_data")
            end
            data_path = data_path * "BAE_results_data/"
        end
    end

    BAE.HP.ϵ = convert(object_type, BAE.HP.ϵ)
    BAE.HP.η = convert(object_type, BAE.HP.η)
    BAE.HP.λ = convert(object_type, BAE.HP.λ)

    Xt = X'
    n = size(X, 1)

    @info "Training BAE for $(BAE.HP.n_restarts) runs with $(BAE.HP.epochs) epochs per run and a batchsize of $(BAE.HP.batchsize), i.e., $(Int(round(BAE.HP.epochs * n / BAE.HP.batchsize))) update iterations per run."

    opt = ADAMW(BAE.HP.η, (0.9, 0.999), BAE.HP.λ)
    opt_state = Flux.setup(opt, BAE.decoder)
    
    mean_trainlossPerEpoch = []
    sparsity_level = []
    entanglement_score = []
    clustering_score = []
    coefficients = []

    for iter in 1:BAE.HP.n_restarts

        @info "Training run: $iter"

        @showprogress for epoch in 1:BAE.HP.epochs

            loader = Flux.Data.DataLoader(Xt, batchsize=BAE.HP.batchsize, shuffle=true) 

            batchLosses = Float32[]
            for batch in loader

                batch_t = batch'
                Z = get_latentRepresentation(BAE, batch)
                
                batchLoss, grads = Flux.withgradient(BAE.decoder, Z) do m, z 
                    X̂ = m(z)
                    Flux.mse(X̂, batch)
                end
                push!(batchLosses, batchLoss)

                disentangled_compL2Boost!(BAE, batch_t, grads[2])
                Flux.update!(opt_state, BAE.decoder, grads[1])

                if track_coeffs && (iter == BAE.HP.n_restarts)
                    push!(coefficients, copy(BAE.coeffs))
                end

            end

            push!(mean_trainlossPerEpoch, mean(batchLosses))
            push!(sparsity_level, 1 - (count(x->x!=0, BAE.coeffs) / length(BAE.coeffs))) # Percentage of zero elements in the encoder weight matrix (higher values are better)
            Z = get_latentRepresentation(BAE, Xt)
            push!(entanglement_score, sum(UpperTriangular(abs.(cor(Z, dims=2)))) - convert(object_type, BAE.HP.zdim)) # Offdiagonal of the Pearson correlation coefficient (upper triangular) matrix between the latent dimensions (closer to 0 is better)
            Z_cluster = softmax(split_vectors(Z))
            push!(clustering_score, n - sum([maximum(Z_cluster[:, i]) for i in 1:n])) # Sum of deviations of the maximum cluster probability value per cell from 1 (closer to 0 is better)

            if save_data #&& epoch % 10 == 0
                file_path = data_path * "/BAE_coeffs.txt"
                writedlm(file_path, BAE.coeffs)
                file_path = data_path * "/Trainloss_BAE.txt"
                writedlm(file_path, mean_trainlossPerEpoch)
                file_path = data_path * "/Sparsity_BAE.txt"
                writedlm(file_path, sparsity_level)
                file_path = data_path * "/Entanglement_BAE.txt"
                writedlm(file_path, entanglement_score)
                file_path = data_path * "/ClusteringScore_BAE.txt"
                writedlm(file_path, clustering_score)
            end

        end

        if iter < BAE.HP.n_restarts
            @info "Re-initialize encoder weights as zeros ..."
            BAE.coeffs = zeros(object_type, size(BAE.coeffs))
        end

    end

    @info "Finished training. Computing results ..."

    BAE.Z = get_latentRepresentation(BAE, Xt)


    # Compute the cluster probabilities for each cell:
    BAE.Z_cluster = softmax(split_vectors(BAE.Z))
    MD.obs_df[!, :Cluster] = [argmax(BAE.Z_cluster[:, i]) for i in 1:size(BAE.Z_cluster, 2)]

    # Compute average Silhouette score of the soft clustering results:
    dist_mat = pairwise(Euclidean(), BAE.Z, dims=2);
    silhouettes = Clustering.silhouettes(MD.obs_df.Cluster, dist_mat)
    MD.obs_df[!, :Silhouettes] = silhouettes

    if isnothing(MD.Top_features)
        MD.Top_features = topFeatures_per_Cluster(BAE, MD; save_data=save_data, data_path=data_path)
    end


    @info "Weight matrix sparsity level after the training: $(100 * count(x->x==0, BAE.coeffs) / length(BAE.coeffs))% (zero values).\n Silhouette score of the soft clustering: $(mean(silhouettes))."

    output_dict = Dict{String, Union{Vector{eltype(X)}, Vector{Any}}}()

    output_dict["trainloss"] = mean_trainlossPerEpoch
    output_dict["sparsity"] = sparsity_level
    output_dict["entanglement"] = entanglement_score
    output_dict["clustering"] = clustering_score
    output_dict["silhouettes"] = silhouettes

    if track_coeffs
        output_dict["coefficients"] = coefficients
    end


    return output_dict
end