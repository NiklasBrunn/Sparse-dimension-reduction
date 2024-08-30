#---Componentwise boosting functions:

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

function compL2Boost!(BAE::BoostingAutoencoder, l::Int, X::AbstractMatrix{T}, y::AbstractVector{T}, denom::AbstractVector{T}) where T
    #determine the number of observations and the number of features in the training data:
    #p = size(X, 2)
    n, p = size(X)

    β = BAE.coeffs[:, l]

    update_inds = findall(x->x!=0, β)

    for step in 1:BAE.HP.M

        #nz_inds = findall(x->x!=0, β)

        #compute the residual as the difference of the target vector and the current fit:
        res = y .- (X * β)

        #determine the p unique univariate OLLS estimators for fitting the residual vector res:
        unibeta = get_unibeta(X, res, denom, n, p) # Currently the faster option
        #unibeta = [dot(X[:, j], res) / denom[j] for j in 1:p] 

        #determine the optimal index of the univariate estimators resulting in the currently optimal fit:
        optindex = findmax(unibeta.^2 .* denom)[2]

        #update β by adding a re-scaled version of the selected OLLS-estimator, by a scalar value ϵ ∈ (0,1):
        update_inds = union(update_inds, optindex)
        BAE.coeffs[update_inds, l] .+= unibeta[update_inds] .* BAE.HP.ϵ  

    end

end

function disentangled_compL2Boost!(BAE::BoostingAutoencoder, batch::AbstractMatrix{T}, grads::AbstractMatrix{T}) where T

    denom = vec(sum(batch .^ 2, dims=1))

    Y = scale(-grads, dims=2)

    for l in 1:BAE.HP.zdim

        #determine the indices of latent dimensions excluded for determining the pseudo-target for boosting:
        Inds = union(find_zero_columns(BAE.coeffs), l)

        if length(Inds) == BAE.HP.zdim
            #since all lat. dims. are excluded, the pseudo target is determined by the st. neg. grad.:
            y = Y[l, :]

            #apply componentwise boosting to update one component of the β-vector(l-th col. of matrix B):
            compL2Boost!(BAE, l, batch, y, denom)

        else
            #compute optimal current OLLS-fit of other latent repr. to the st. neg. grad:
            curdata = batch * BAE.coeffs[:, Not(Inds)]
            curtarget = Y[l, :]
            
            # Solve the least squares problem using Cholesky decomposition
            m = size(curdata, 2)
            Λ = convert(eltype(batch), 1.0e-6) .* Matrix(I, m, m) # Regularization term that prevents singular matrices (rarely happens)
            XtX = curdata' * curdata .+ Λ
            XtY = curdata' * curtarget
            
            # Use Cholesky decomposition to solve the linear system XtX * curestimate = XtY
            curestimate = ldiv!(cholesky(XtX, Val(true)), XtY)

            #compute the pseudo-target for the boosting [st. difference of st. neg. grad and optimal OLLS-fit]:
            y = scale(curtarget .- (curdata * curestimate))

            #apply componentwise boosting to update one component of the β-vector (l-th col. of matrix B):
            compL2Boost!(BAE, l, batch, y, denom)
        end
        
    end

end

#---training function for a BAE with the disentanglement constraint:
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
            mkdir(data_path * "BAE_results_data")
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

    # Compute average Silhouette score and Adjusted Rand index:
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