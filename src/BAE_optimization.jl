#---componentwise boosting functions:
function calcunibeta(X::AbstractMatrix{<:AbstractFloat}, res::AbstractVector{<:AbstractFloat}, n::Int, p::Int)
    unibeta = zeros(p)
    denom = zeros(p)

    #compute the univariate OLLS-estimator for each component 1:p:
    for j = 1:p

       for i=1:n
          unibeta[j] += X[i, j]*res[i]
          denom[j] += X[i, j]*X[i, j]
       end

       unibeta[j] /= denom[j] 

    end

    #return a vector unibeta consisting of the OLLS-estimators and another vector, 
    #consisting of the denominators (for later re-scaling)
    return unibeta, denom 
end

function compL2Boost!(BAE::BoostingAutoencoder, l::Int, X::AbstractMatrix{<:AbstractFloat}, y::AbstractVector{<:Number})
    #determine the number of observations and the number of features in the training data:
    n, p = size(X)

    for step in 1:BAE.HP.M

        nz_inds = findall(x->x!=0, BAE.coeffs[:, l])

        #compute the residual as the difference of the target vector and the current fit:
        curmodel = X * BAE.coeffs[:, l]
        res = y .- curmodel

        #determine the p unique univariate OLLS estimators for fitting the residual vector res:
        unibeta, denom = calcunibeta(X, res, n, p) 

        #determine the optimal index of the univariate estimators resulting in the currently optimal fit:
        optindex = findmax(collect(unibeta[j]^2 * denom[j] for j in 1:p))[2]

        #update β by adding a re-scaled version of the selected OLLS-estimator, by a scalar value ϵ ∈ (0,1):
        BAE.coeffs[union(nz_inds, optindex), l] .+= unibeta[union(nz_inds, optindex)] .* BAE.HP.ϵ  

    end

end

function disentangled_compL2Boost!(BAE::BoostingAutoencoder, batch::AbstractMatrix{<:AbstractFloat}, grads::AbstractMatrix{<:AbstractFloat})

    for l in 1:BAE.HP.zdim

        #determine the indices of latent dimensions excluded for determining the pseudo-target for boosting:
        Inds = union(find_zero_columns(BAE.coeffs), l)

        if length(Inds) == BAE.HP.zdim
            #since all lat. dims. are excluded, the pseudo target is determined by the st. neg. grad.:
            y = scale(-grads[l, :])

            #apply componentwise boosting to update one component of the β-vector(l-th col. of matrix B):
            compL2Boost!(BAE, l, batch, y)

        else
            #compute optimal current OLLS-fit of other latent repr. to the st. neg. grad:
            curdata = batch * BAE.coeffs[:, Not(Inds)]
            curtarget = scale(-grads[l, :]) 
            curestimate = inv(curdata'curdata)*(curdata'curtarget) 
             #curestimate = inv(curdata'curdata + Float32.(1.0e-5 * Matrix(I, size(curdata, 2), size(curdata, 2))))*(curdata'curtarget)  #damped version for avoiding invertibility problems

            #compute the pseudo-target for the boosting [st. difference of st. neg. grad and optimal OLLS-fit]:
            y = scale(curtarget .- (curdata * curestimate))

            #apply componentwise boosting to update one component of the β-vector (l-th col. of matrix B):
            compL2Boost!(BAE, l, batch, y)
        end
        
    end

end

#---training function for a BAE with the disentanglement constraint:
function train_BAE!(X::AbstractMatrix{<:AbstractFloat}, BAE::BoostingAutoencoder; soft_clustering::Bool=false, MD::Union{Nothing, MetaData}=nothing, save_data::Bool=false, path::Union{Nothing, String}=nothing, batchseed::Int=42)

    Random.seed!(batchseed)

    X = Float32.(X)
    Xt = X'
    n = size(X, 1)

    @info "Training BAE for $(BAE.HP.epochs) epochs with a batchsize of $(BAE.HP.batchsize), i.e., $(Int(round(BAE.HP.epochs * n / BAE.HP.batchsize))) update iterations."

    opt = ADAMW(BAE.HP.η, (0.9, 0.999), BAE.HP.λ)
    opt_state = Flux.setup(opt, BAE.decoder)
    
    mean_trainlossPerEpoch = []
    sparsity_level = []
    entanglement_score = []
    clustering_score = []
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
        end

        push!(mean_trainlossPerEpoch, mean(batchLosses))
        push!(sparsity_level, 1 - (count(x->x!=0, BAE.coeffs) / length(BAE.coeffs))) # Percentage of nonzero elements in the encoder weight matrix (lower values are better)
        Z = get_latentRepresentation(BAE, Xt)
        push!(entanglement_score, sum(UpperTriangular(abs.(cor(Z, dims=2)))) - BAE.HP.zdim) # Offdiagonal of the Pearson correlation coefficient (upper triangular) matrix between the latent dimensions (closer to 0 is better)
        Z_cluster = softmax(split_vectors(Z))
        push!(clustering_score, n - sum([maximum(Z_cluster[:, i]) for i in 1:n])) # Sum of deviations of the maximum cluster probability value per cell from 1 (closer to 0 is better)

    end

    BAE.Z = get_latentRepresentation(BAE, Xt)

    if soft_clustering

        # Compute the cluster probabilities for each cell:
        BAE.Z_cluster = softmax(split_vectors(BAE.Z))
        MD.obs_df[!, :Cluster] = [argmax(BAE.Z_cluster[:, i]) for i in 1:size(BAE.Z_cluster, 2)]

        # Compute average Silhouette score and Adjusted Rand index:
        dist_mat = pairwise(Euclidean(), BAE.Z, dims=2);
        silhouettes = Clustering.silhouettes(MD.obs_df.Cluster, dist_mat)
        MD.obs_df[!, :Silhouettes] = silhouettes

        if !isnothing(MD.Top_features)
            MD.Top_features = topFeatures_per_Cluster(BAE; save_data=save_data, path=path)
        end

        @info "Weight matrix sparsity level after the training: $(100 * count(x->x!=0, BAE.coeffs) / length(BAE.coeffs))% (nonzero values).\n Silhouette score of the clustering: $(mean(silhouettes))."
    
    else

        @info "Weight matrix sparsity level after the training: $(100 * count(x->x!=0, BAE.coeffs) / length(BAE.coeffs))% (nonzero values)."

    end

    return mean_trainlossPerEpoch, sparsity_level, entanglement_score, clustering_score, silhouettes
end