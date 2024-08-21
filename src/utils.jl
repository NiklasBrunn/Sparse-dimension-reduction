#---Utils:

# Data transformation:
function scale(X::AbstractArray{T}; corrected_std::Bool=true, dims::Int=1) where T
    return (X .- mean(X, dims=dims)) ./ std(X, corrected=corrected_std, dims=dims)
end

function NaNstoZeros!(X::AbstractArray{T}) where T

    X[isnan.(X)] .= convert(eltype(X), 0)

    return X
end

function Normalize(X::AbstractMatrix{T}; dims::Int=2, rescale::Union{Int, AbstractFloat}=10000) where T

    if dims == 1
        @info "Normalizing matrix over the rows"
    else
        @info "Normalizing matrix over the colums"
    end

    Z=zeros(eltype(X), size(X))

    if dims == 2
        Z = X./vec(sum(X, dims=dims))
    else
        Z = transpose(X'./vec(sum(X, dims=dims)))
    end

    rescale = convert(eltype(X), rescale)
    Z.*=rescale

    return Matrix(Z)
end

function MinMaxNormalize(v::AbstractVector{T}) where T
    return (v .- minimum(v)) ./ (maximum(v) - minimum(v))
end

function log1p(X::AbstractArray{T}) where T
    return log.(1 .+ X)
end

function log1pNormalize(X::AbstractArray{T}; dims::Int=2, rescale::Union{Int, AbstractFloat}=10000) where T

    Z = Normalize(X; dims=dims, rescale=rescale)
    Z = log1p(Z)

    return Z
end

function find_zero_columns(X::AbstractMatrix{T}) where T
    v = vec(sum(abs.(X), dims=1))
    zero_cols = findall(x->x==0, v)
    return zero_cols
end

function split_vectors(Z::Matrix{T}) where T
    d, n = size(Z)
    Z_new = ones(Float32, n)
    count = 1
    for l in 1:2*d
        if l % 2 == 1
            Z_new = hcat(Z_new, Z[count, :])
        else
            Z_new = hcat(Z_new, -Z[count, :])
            count+=1
        end
    end

    return Z_new[:, 2:end]'
end


# Data filtering (Interaction data and gene expression data where observations are cell pairs or cells and features are interactions or genes):
function filter_observations(X::Matrix{T}; min_features::Union{Int, Nothing}=10, min_expression_threshold::Union{T, Nothing}=nothing, max_expression_threshold::Union{T, Nothing}=nothing) where T

    filtered_data = []

    if !isnothing(min_features) && !isnothing(min_expression_threshold) && !isnothing(max_expression_threshold)
        # Compute the number of expressed genes (non-zero values) per cell
        expressed_genes_count = vec(sum(X .> 0, dims=2))

        # Compute the total expression per cell
        total_expression = vec(sum(X, dims=2))

        # Filter cells based on the criteria
        filtered_indices = findall((expressed_genes_count .>= min_features) .& (total_expression .>= min_expression_threshold) .& (total_expression .<= max_expression_threshold))

        @info "$(size(X, 1) - length(filtered_indices)) observations were excluded based on the filtering criterion."

    elseif !isnothing(min_features) && isnothing(min_expression_threshold) && isnothing(max_expression_threshold)
        # Compute the number of expressed genes (non-zero values) per cell
        expressed_genes_count = vec(sum(X .> 0, dims=2))

        # Filter cells based on the criteria
        filtered_indices = findall(expressed_genes_count .>= min_features)

        @info "$(size(X, 1) - length(filtered_indices)) observations were excluded based on the filtering criterion."

    elseif isnothing(min_features) && !isnothing(min_expression_threshold) && !isnothing(max_expression_threshold)
        # Compute the total expression per cell
        total_expression = vec(sum(X, dims=2))

        # Filter cells based on the criteria
        filtered_indices = findall((total_expression .>= min_expression_threshold) .& (total_expression .<= max_expression_threshold))

        @info "$(size(X, 1) - length(filtered_indices)) observations were excluded based on the filtering criterion."

    else
        error("You must provide at least one of the filtering criteria.")
    end
    
    # Select only the cells that meet the criteria
    filtered_data = X[filtered_indices, :]
    
    return filtered_data, filtered_indices
end

function filter_features(X::Matrix{T}; min_obs::Union{Int, Nothing}=10, min_expression_threshold::Union{T, Nothing}=nothing, max_expression_threshold::Union{T, Nothing}=nothing) where T

    filtered_data = []

    if !isnothing(min_obs) && !isnothing(min_expression_threshold) && !isnothing(max_expression_threshold)
        # Compute the number of cells where the gene is expressed (non-zero values)
        expressed_cells_count = vec(sum(X .> 0, dims=1))

        # Compute the total expression per gene across all cells
        total_expression = vec(sum(X, dims=1))

        # Filter genes based on the criteria
        filtered_indices = findall((expressed_cells_count .>= min_obs) .& (total_expression .>= min_expression_threshold) .& (total_expression .<= max_expression_threshold))

        @info "$(size(X, 2) - length(filtered_indices)) features were excluded based on the filtering criterion."

    elseif !isnothing(min_obs) && isnothing(min_expression_threshold) && isnothing(max_expression_threshold)
        # Compute the number of cells where the gene is expressed (non-zero values)
        expressed_cells_count = vec(sum(X .> 0, dims=1))

        # Filter genes based on the criteria
        filtered_indices = findall(expressed_cells_count .>= min_obs)

        @info "$(size(X, 2) - length(filtered_indices)) features were excluded based on the filtering criterion."

    elseif isnothing(min_obs) && !isnothing(min_expression_threshold) && !isnothing(max_expression_threshold)
        # Compute the total expression per gene across all cells
        total_expression = vec(sum(X, dims=1))

        # Filter genes based on the criteria
        filtered_indices = findall((total_expression .>= min_expression_threshold) .& (total_expression .<= max_expression_threshold))

        @info "$(size(X, 2) - length(filtered_indices)) features were excluded based on the filtering criterion."

    else
        error("You must provide at least one of the filtering criteria.")
    end
    
    # Select only the genes that meet the criteria
    filtered_data = X[:, filtered_indices]
    
    return filtered_data, filtered_indices
end

function filter_cells_by_mitochondrial_content(X::Matrix{T}, genenames::Vector{String}; 
    species::String="human", mito_threshold::Float64=0.15) where T

    # Define mitochondrial gene prefixes for human and mouse
    mitochondrial_prefix = if species == "human"
        "MT-"  # Mitochondrial genes in humans typically start with "MT-"
    elseif species == "mouse"
        "mt-"  # Mitochondrial genes in mice typically start with "mt-"
    else
        error("Unsupported species. Please specify 'human' or 'mouse'.")
    end

    # Identify mitochondrial genes
    is_mitochondrial = startswith.(genenames, mitochondrial_prefix)

    # Calculate the proportion of mitochondrial gene expression per cell
    mitochondrial_counts = vec(sum(X[:, is_mitochondrial], dims=2))
    total_counts = vec(sum(X, dims=2))
    mito_proportion = mitochondrial_counts ./ total_counts

    # Filter out cells with high mitochondrial content
    filtered_indices = findall(mito_proportion .<= mito_threshold)
    
    # Select only the cells that meet the criteria
    filtered_data = X[filtered_indices, :]
    
    @info "$(size(X, 1) - length(filtered_indices)) cells were excluded based on mitochondrial content."
    
    return filtered_data, filtered_indices
end


# Generate BAE results:
function get_latentRepresentation(BAE::BoostingAutoencoder, X::AbstractMatrix{T}) where T
    Z = transpose(BAE.coeffs) * X
    return Z
end

function topFeatures_per_Cluster(BAE::BoostingAutoencoder, MD::MetaData; save_data::Bool=true, path::Union{Nothing, String}=nothing)

    TopFeatures_Cluster = Dict{String, DataFrame}();
    counter_var = 1;
    for l in 1:size(BAE.coeffs, 2)
        pos_inds = findall(x->x>0, BAE.coeffs[:, l])

        if isempty(pos_inds)
            df_pos = DataFrame(Features=[])
            df_pos[!, "Scores"] = []
            df_pos[!, "normScores"] = []
            TopFeatures_Cluster["$(counter_var)"] = df_pos
        else
            df_pos = DataFrame(Features=MD.featurename[pos_inds])
            df_pos[!, "Scores"] = BAE.coeffs[pos_inds, l]
            df_pos[!, "normScores"] = df_pos[!, "Scores"] ./ maximum(df_pos[!, "Scores"])
            sort!(df_pos, :Scores, rev=true)
            TopFeatures_Cluster["$(counter_var)"] = df_pos
        end

        counter_var+=1

        neg_inds = findall(x->x>0, -BAE.coeffs[:, l])
        
        if isempty(neg_inds)
            df_neg = DataFrame(Features=[])
            df_neg[!, "Scores"] = []
            df_neg[!, "normScores"] = []
            TopFeatures_Cluster["$(counter_var)"] = df_neg
        else
            df_neg = DataFrame(Features=MD.featurename[neg_inds])
            df_neg[!, "Scores"] = -BAE.coeffs[neg_inds, l]
            df_neg[!, "normScores"] = df_neg[!, "Scores"] ./ maximum(df_neg[!, "Scores"])
            sort!(df_neg, :Scores, rev=true)
            TopFeatures_Cluster["$(counter_var)"] = df_neg
        end

        counter_var+=1
    end

    if save_data && !isnothing(path) 
        if !isdir(path * "/TopFeaturesCluster_CSV")
            # Create the folder if it does not exist
            mkdir(path * "/TopFeaturesCluster_CSV")
        end

        for (key, df) in TopFeatures_Cluster
            filepath = joinpath(path * "/TopFeaturesCluster_CSV/", "topFeatures_Cluster_" * key * ".csv")
            CSV.write(filepath, df)
        end
    end
    
    return TopFeatures_Cluster
end


# Visualization:
function generate_umap(X::AbstractMatrix{T}, seed::Int; n_neighbors::Int=30, min_dist::Float64=0.4) where T
    Random.seed!(seed)  
    embedding = umap(X', n_neighbors=n_neighbors, min_dist=min_dist)'

    return embedding
end

function hsl_to_hex(h, s, l)
    if s == 0
        # Achromatic, i.e., grey
        hex = lpad(string(round(Int, l * 255), base=16), 2, '0')
        return "#$hex$hex$hex"
    end

    function hue_to_rgb(p, q, t)
        if t < 0 
            t += 1 
        end
        if t > 1 
            t -= 1 
        end
        if t < 1/6
            return p + (q - p) * 6 * t
        elseif t < 1/2
            return q
        elseif t < 2/3
            return p + (q - p) * (2/3 - t) * 6
        else
            return p
        end
    end

    q = l < 0.5 ? l * (1 + s) : l + s - l * s
    p = 2 * l - q

    r = hue_to_rgb(p, q, h + 1/3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1/3)

    hex_r = lpad(string(round(Int, r * 255), base=16), 2, '0')
    hex_g = lpad(string(round(Int, g * 255), base=16), 2, '0')
    hex_b = lpad(string(round(Int, b * 255), base=16), 2, '0')

    return "#$hex_r$hex_g$hex_b"
end