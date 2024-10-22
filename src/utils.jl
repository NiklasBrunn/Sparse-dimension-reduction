#---Utils:

# Data transformation:
#function scale(X::AbstractArray{T}; corrected_std::Bool=true, dims::Int=1) where T
#    return (X .- mean(X, dims=dims)) ./ std(X, corrected=corrected_std, dims=dims)
#end

"""
    scale(X::AbstractArray{T}; corrected_std::Bool=true, dims::Int=1) where T

Scales the input array `X` by centering and normalizing along the specified dimension(s).

This function standardizes the array `X` by subtracting the mean and dividing by the standard deviation along the specified dimension `dims`. If `corrected_std` is `true`, the corrected (unbiased) standard deviation is used.

# Arguments
- `X::AbstractArray{T}`: Input array to be standardized. Can be of any numeric type `T`.
- `corrected_std::Bool=true`: Whether to use the corrected (unbiased) standard deviation, which divides by `n-1`. If `false`, divides by `n` (default: `true`).
- `dims::Int=1`: Dimension along which to compute the mean and standard deviation. Defaults to `1` (i.e., along columns for matrices).

# Returns
- A new array of the same type and shape as `X`, where each element is standardized (mean = 0, standard deviation = 1) along the specified dimension.

The operation is performed in a memory-efficient way using views and in-place broadcasting.
"""
@inline function scale(X::AbstractArray{T}; corrected_std::Bool=true, dims::Int=1) where T
    M = mean(X, dims=dims)
    S = std(X, corrected=corrected_std, dims=dims)
    X_st = similar(X)
    @inbounds @views X_st .= (X .- M) ./ S
    return X_st
end

"""
    NaNstoZeros!(X::AbstractArray{T}) where T

Replaces all `NaN` values in the input array `X` with zeros, modifying the array in place.

This function searches for `NaN` values in the array `X` and replaces them with zeros of the appropriate type, determined by the element type of `X`.

# Arguments
- `X::AbstractArray{T}`: Input array of any numeric type `T` that may contain `NaN` values. The array is modified in place.

# Returns
- The input array `X`, with all `NaN` values replaced by zeros, returned as the same array object (in-place operation).

# Example
```julia
X = [1.0, NaN, 3.0]
NaNstoZeros!(X)  # X becomes [1.0, 0.0, 3.0]
```
"""
function NaNstoZeros!(X::AbstractArray{T}) where T

    X[isnan.(X)] .= convert(eltype(X), 0)

    return X
end

"""
    Normalize(X::AbstractMatrix{T}; dims::Int=2, rescale::Union{Int, AbstractFloat}=10000) where T

Normalizes the input matrix `X` along the specified dimension and rescales the values by a factor.

This function normalizes the matrix `X` by dividing each element by the sum of elements along the specified dimension `dims`. After normalization, the resulting matrix is scaled by the `rescale` factor. The function handles row-wise or column-wise normalization based on the value of `dims`.

# Arguments
- `X::AbstractMatrix{T}`: The input matrix of numeric type `T` to be normalized.
- `dims::Int=2`: The dimension along which to normalize. `dims=1` normalizes over rows, and `dims=2` (default) normalizes over columns.
- `rescale::Union{Int, AbstractFloat}=10000`: The scaling factor to apply after normalization. Defaults to `10000`.

# Returns
- A new matrix of the same type as `X`, where the values are normalized and rescaled.

# Example
```julia
X = [1.0 2.0; 3.0 4.0]
Y = Normalize(X, dims=2, rescale=10000)  # Normalizes over columns and scales the result
```
"""
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

"""
    MinMaxNormalize(v::AbstractVector{T}) where T

Applies min-max normalization to the input vector `v`.

This function scales the elements of the vector `v` such that the minimum value in `v` is mapped to 0, and the maximum value is mapped to 1. All other elements are scaled proportionally within this range.

# Arguments
- `v::AbstractVector{T}`: A vector of any numeric type `T`.

# Returns
A vector of the same size as `v`, with elements scaled to the range [0, 1].

# Example
```julia
v = [1, 2, 3, 4, 5]
normalized_v = MinMaxNormalize(v)
# Result: [0.0, 0.25, 0.5, 0.75, 1.0]
```
"""
function MinMaxNormalize(v::AbstractVector{T}) where T
    return (v .- minimum(v)) ./ (maximum(v) - minimum(v))
end

"""
    log1p(X::AbstractArray{T}) where T

Applies the element-wise transformation `log(1 + x)` to each element in the array `X`, with `1` converted to the element type of `X` for type stability.

# Arguments
- `X::AbstractArray{T}`: An array of any numeric type `T`.

# Returns
An array of the same size and type as `X`, where each element is the result of applying the `log(1 + x)` operation with `1` converted to the type of `X`.

# Example
```julia
X = [0.0, 1.0, 2.0]
result = log1p(X)
# Result: [0.0, 0.693147, 1.098612]
```
"""
function log1p(X::AbstractArray{T}) where T
    return log.(convert(eltype(X), 1) .+ X)
end

"""
    log1pNormalize(X::AbstractArray{T}; dims::Int=2, rescale::Union{Int, AbstractFloat}=10000) where T

Applies normalization followed by the `log(1 + x)` transformation to the input array `X`.

This function first normalizes the input array `X` along the specified dimension using the provided rescaling factor. It then applies the element-wise `log(1 + x)` transformation to the normalized array. The normalization step rescales the values in `X` based on the specified `dims` and `rescale` values.

# Arguments
- `X::AbstractArray{T}`: An array of any numeric type `T`.
- `dims::Int=2`: The dimension along which normalization is applied. Defaults to `2`.
- `rescale::Union{Int, AbstractFloat}=10000`: A rescaling factor applied during normalization. Defaults to `10000`.

# Returns
An array of the same size and type as `X`, where each element is first normalized and then transformed using `log(1 + x)`.

# Example
```julia
X = [0.0 2.0; 3.0 4.0]
Z = log1pNormalize(X, dims=1, rescale=10000)
# Result: A normalized array with log1p transformation applied
```
"""
function log1pNormalize(X::AbstractArray{T}; dims::Int=2, rescale::Union{Int, AbstractFloat}=10000) where T

    Z = Normalize(X; dims=dims, rescale=rescale)
    Z = log1p(Z)

    return Z
end

function find_zero_columns(X::AbstractMatrix{T}) where T
    v = vec(sum(abs.(X), dims=1))
    zero_cols = findall(x->x==convert(eltype(X), 0), v)
    return zero_cols
end

@inline function split_vectors(Z::Matrix{T}) where T
    d, n = size(Z)
    Z_new = ones(eltype(Z), n)
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
@inline function get_latentRepresentation(BAE::BoostingAutoencoder, X::AbstractMatrix{T}) where T
    @inbounds Z = transpose(BAE.coeffs) * X
    return Z
end

function topFeatures_per_Cluster(BAE::BoostingAutoencoder, MD::MetaData; save_data::Bool=true, data_path::Union{Nothing, String}=nothing)

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

    if save_data && !isnothing(data_path) 
        if !isdir(data_path * "/TopFeaturesCluster_CSV")
            # Create the folder if it does not exist
            mkdir(data_path * "/TopFeaturesCluster_CSV")
        end

        for (key, df) in TopFeatures_Cluster
            filepath = joinpath(data_path * "/TopFeaturesCluster_CSV/", "topFeatures_Cluster_" * key * ".csv")
            CSV.write(filepath, df)
        end
    end
    
    return TopFeatures_Cluster
end

function get_important_genes(BAE::BoostingAutoencoder, MD::MetaData; save_data::Bool=false, data_path::Union{String, Nothing}=nothing)

    important_genes_inds = findall(x->x!=convert(eltype(BAE.coeffs), 0), vec(sum(abs.(BAE.coeffs), dims=2)))
    important_genes = MD.featurename[important_genes_inds]

    if save_data
        writedlm(data_path * "important_genes.txt", important_genes)
    end
    
    return important_genes
end

function get_topFeatures(BAE::BoostingAutoencoder, MD::MetaData)
    #Currently per latent dimension. Should it be updated such that top features per cluster are computed?

    if isnothing(MD.featurename)
        @warn "The metadata does not contain the feature names ..."
        MD.featurename = ["$(j)" for j in 1:size(BAE.coeffs, 1)]
    end

    dict = Dict{Int, Vector{String}}();

    for l in 1:size(BAE.coeffs, 2)

        weights = abs.(BAE.coeffs[:, l])
        inds = sortperm(weights; rev=true)
        weights = weights[inds]
        sort_featurenames = MD.featurename[inds]

        nonzero_inds = findall(x->x!=0, weights)
        weights = weights[nonzero_inds]
        sort_featurenames =  sort_featurenames[nonzero_inds]

        diffs = [weights[i] - weights[i+1] for i in 1:length(weights)-1]
        topGene_inds = findmax(diffs)[2]      
        sort_featurenames[1:topGene_inds]

        dict[l] = sort_featurenames[1:topGene_inds]
    end

    return dict
end


# Visualization:
function generate_umap(X::AbstractMatrix{T}; n_neighbors::Int=30, min_dist::Float64=0.4, seed::Int=42) where T
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


# Tutorial notebook:
function load_rat_scRNAseq_data(; 
    max_time::Int=600, 
    data_path::Union{String, Nothing}=nothing, 
    subset_data::Bool=true, 
    transfer_data::Bool=false, 
    assay::Union{String, Nothing}="RNA"
    )

    if !isnothing(data_path) && !isdir(data_path)
        @error "The data directory does not exist. Please create the directory or provide a valid path. Otherwise, set the data_path to nothing to automate the creation of a directory."
    end

    file_path = []
    download_data =true

    if subset_data
        if isnothing(data_path)

            projectpath = joinpath(@__DIR__, "../")
            data_path = projectpath * "data/tutorial/"
            if !isdir(data_path)
                mkdir(data_path)
            end

        end

        file_path = data_path * "Rat_Seurat_sub.rds"
        if isfile(file_path)
            @warn "The file already exists. Skipping the download ..."
            download_data = false
        else
            println("Loading the subsetted rat lung data (~ 1.6 GB) this takes a few minutes ... .")
            println("The data will be saved to: $(data_path)")
        end

    else
        if isnothing(data_path)

            projectpath = joinpath(@__DIR__, "../")
            data_path = projectpath * "data/tutorial/"
            if !isdir(data_path)
                mkdir(data_path)
            end

        end

        file_path = data_path * "Rat_Seurat.rds"
        if isfile(file_path)
            @warn "The file already exists. Skipping the download ..."
            download_data = false
        else
            println("Loading the rat lung data (~ 2.7 GB) this takes a few minutes ... .")
            println("The data will be saved to: $(data_path)")
        end
    end


    R"""
    library(Seurat)
    library(NICHES)
    """

    if download_data && subset_data
        @rput data_path
        @rput max_time

        R"""
        options(timeout = max(max_time, getOption("timeout"))) #set download time limit to 12 minutes ...
        url <- "https://zenodo.org/record/6846618/files/raredon_2019_rat.Robj"
        temp_file <- tempfile(fileext = ".Robj")
        download.file(url, destfile = temp_file, mode = "wb")
        load(temp_file)

        rat.sub <- subset(rat, idents = c('ATII','ATI', 'Mac_alv','Mac_inter',
                                          'Fib_Col13a1+','Fib_Col14a1+','SMCs', 
                                          'EC_cap','EC_vasc')
        )

        rat.sub@meta.data$cell_types <- Idents(rat.sub)

        saveRDS(rat.sub, file = paste0(data_path, "Rat_Seurat_sub.rds"))

        rm(list=ls())
        gc()
        """

    elseif download_data && !subset_data
        @rput data_path
        @rput max_time

        R"""
        options(timeout = max(max_time, getOption("timeout"))) #set download time limit to 12 minutes ...
        url <- "https://zenodo.org/record/6846618/files/raredon_2019_rat.Robj"
        temp_file <- tempfile(fileext = ".Robj")
        download.file(url, destfile = temp_file, mode = "wb")
        load(temp_file)

        rat@meta.data$cell_types <- Idents(rat)

        saveRDS(rat, file = paste0(data_path, "Rat_Seurat.rds"))

        rm(list=ls())
        gc()
        """
        
    end


    if transfer_data
        println("Transferring the data to Julia ...")

        @rput file_path

        R"""
        seurat_obj <- readRDS(file_path)

        celltype <- Idents(seurat_obj)

        genename <- rownames(seurat_obj)

        X <- matrix(nrow = 0, ncol = 0)

        embedding <- Embeddings(seurat_obj, "tsne")
        """

        if isnothing(assay)
            assay = "RNA"

            @rput assay

            R"""
            X <- as.matrix(GetAssayData(object = seurat_obj, assay = assay, slot = "counts"))
            """
        elseif assay == "alra"
            @rput assay

            R"""
            X <- as.matrix(GetAssayData(object = seurat_obj, assay = assay, slot = "data"))
            """
        elseif assay == "RNA"
            @rput assay

            R"""
            X <- as.matrix(GetAssayData(object = seurat_obj, assay = assay, slot = "counts"))
            """
        else
            error("The data layer is not supported. Please provide a valid data layer.")
        end

        @rget X
        @rget embedding
        @rget celltype
        @rget genename

        X = Matrix{Float32}(X')
        embedding = Matrix{Float32}(embedding)
        celltype = String.(celltype)
        genename = String.(genename)

        R"""
        rm(list=ls())
        gc()
        """

        println("Data transfer completed.")

        return X, embedding, celltype, genename
    end

end

function load_spatial_mousebrain_data(; 
    data_path::Union{String, Nothing}=nothing
    )

    if !isnothing(data_path) && !isdir(data_path)
        @error "The data directory does not exist. Please create the directory or provide a valid path. Otherwise, set the data_path to nothing to automate the creation of a directory."
    end

    file_path = []
    download_data =true

    if isnothing(data_path)

        projectpath = joinpath(@__DIR__, "../")
        data_path = projectpath * "data/tutorial/"
        if !isdir(data_path)
            mkdir(data_path)
        end

        println("Loading the Visium mouse brain data (~ 209 MB), this takes a few seconds ...")
        println("The data will be saved to: $(data_path)")

    else 
        println("Loading the subsetted rat lung data (~ 209 MB), this takes a few seconds ...")
        println("The data will be saved to: $(data_path)")

    end

    file_path = data_path * "MouseBrain_Seurat.rds"
    if isfile(file_path)
        @warn "The file already exists. Skipping the download ..."
        download_data = false
    end

    if download_data
        @rput data_path

        R"""
        library(Seurat)
        library(SeuratData)

        # Load the spatial mous brain data:
        InstallData("stxBrain")
        brain <- LoadData("stxBrain", type = "anterior1")

        # Update seurat object:
        brain <- UpdateSeuratObject(brain) 

        # Normalization: 
        brain <- SCTransform(brain, assay = "Spatial", verbose = FALSE)

        # Run PCA, Find Neighbors, Find Clusters, and Run UMAP:
        brain <- RunPCA(brain, assay = "SCT", verbose = FALSE)
        brain <- FindNeighbors(brain, reduction = "pca", dims = 1:30)
        brain <- FindClusters(brain, verbose = FALSE)
        seurat_clusters <- brain@meta.data$seurat_clusters
        brain <- RunUMAP(brain, reduction = "pca", dims = 1:30)

        # Add spatial coordinates to the metadata:
        x_coords <- brain@images$anterior1@boundaries$centroids@coords[, 1]
        y_coords <- brain@images$anterior1@boundaries$centroids@coords[, 2]
        brain@meta.data$x <- x_coords
        brain@meta.data$y <- y_coords

        # Set the default assay to spatial and normalize counts:
        DefaultAssay(brain) <- "Spatial"
        brain <- NormalizeData(brain)

        # ALRA imputation:
        brain <- SeuratWrappers::RunALRA(brain, verbose = F)
        X <- as.matrix(GetAssayData(object = brain, assay = "alra", slot = "data"))
        genenames <- rownames(brain)

        saveRDS(brain, file = paste0(data_path, "MouseBrain_Seurat.rds"))
        """
        @rget x_coords
        @rget y_coords
        @rget seurat_clusters
        @rget X
        @rget genenames

        X = Matrix{Float32}(X')
        genenames = String.(genenames)

        cell_locations = Matrix{Float32}(hcat(x_coords, y_coords))
        seurat_clusters = String.(seurat_clusters)

        R"""
        rm(list=ls())
        gc()
        """

        return X, cell_locations, seurat_clusters, genenames

    else
        @rput data_path

        R"""
        library(Seurat)

        brain <- readRDS(paste0(data_path, "MouseBrain_Seurat.rds"))

        X <- as.matrix(GetAssayData(object = brain, assay = "alra", slot = "data"))
        genenames <- rownames(brain)

        x_coords <- brain@meta.data$x
        y_coords <- brain@meta.data$y
        seurat_clusters <- brain@meta.data$seurat_clusters
        """

        @rget x_coords
        @rget y_coords
        @rget seurat_clusters
        @rget X
        @rget genenames

        X = Matrix{Float32}(X')
        genenames = String.(genenames)

        cell_locations = Matrix{Float32}(hcat(x_coords, y_coords))
        seurat_clusters = String.(seurat_clusters)

        R"""
        rm(list=ls())
        gc()
        """

        return X, cell_locations, seurat_clusters, genenames

    end


end

function create_seurat_object(X::AbstractMatrix{T}, MD::MetaData; 
    data_path::Union{String, Nothing}=nothing, 
    file_name::Union{String, Nothing}=nothing,
    assay::Union{String, Nothing}= "RNA",
    normalize_data::Bool=true, 
    alra_imputation::Bool=false,
    indents::Union{String, Nothing}=nothing,
    data_is_normalized::Bool=true
    ) where T

    #---Check if the data path exists:
    if isnothing(data_path)

        projectpath = joinpath(@__DIR__, "../")
        data_path = projectpath * "data/tutorial/"
        if !isdir(data_path)
            mkdir(data_path)
            @info "No data path was provided. Creating a directory: $(data_path)"
        end

        @info "No data path was provided. Setting directory to: $(data_path)"
    end

    if isnothing(file_name)

        file_name = data_path * "Seurat_object.rds"

        @info "No file name was provided. Setting the file name to: Seurat_object.rds"
    end

    set_indents = false
    if !isnothing(indents) && indents in names(MD.obs_df)
        set_indents = true
    elseif !isnothing(indents)
        @error "The provided indent column does not exist in the metadata. Please choose an existing column."
    end

    if isnothing(assay)
        assay = "RNA"
        @info "No assay provided. Setting assay to RNA."
    end

    genenames = MD.featurename
    metadata = MD.obs_df

    R"library(Seurat)"

    @rput data_path
    @rput X
    @rput genenames
    @rput normalize_data
    @rput alra_imputation
    @rput metadata
    @rput set_indents
    @rput indents
    @rput assay
    @rput data_is_normalized

    # Create a Seurat object:
    R"""
    seurat_obj <- CreateSeuratObject(counts = X, assay = assay)
    rownames(seurat_obj) <- genenames

    rownames(metadata) <- colnames(seurat_obj)
    seurat_obj <- AddMetaData(object = seurat_obj, metadata = metadata)
    """

    R"""
    if (set_indents) {
        Idents(seurat_obj) <- seurat_obj@meta.data[[indents]]
    }
    """

    R"""
    if (data_is_normalized) {
        LayerData(seurat_obj, assay = assay, layer = "data") <- LayerData(seurat_obj, assay = assay, layer = "counts")
    }
    """

    R"""
    #---Normalize the data:
    if (normalize_data) {
        print("Normalizing data ...")
        seurat_obj <- NormalizeData(seurat_obj)
    }
    """

    #---Impute the data:
    R"""
    if (alra_imputation) {
        print("Performing ALRA imputation ...")
        seurat_obj <- SeuratWrappers::RunALRA(seurat_obj, verbose = F)
    }
    """
    
    # Saving the Seurat object:
    R"""
    saveRDS(seurat_obj, file = paste0(data_path, "Seurat_object.rds"))
    """

    # Remove all objects from the R workspace:
    R"""
    rm(list=ls())
    gc()
    """
end

function create_seurat_object(df::DataFrame, MD::MetaData; 
    data_path::Union{String, Nothing}=nothing, 
    file_name::Union{String, Nothing}=nothing,
    assay::Union{String, Nothing}= "RNA",
    normalize_data::Bool=true, 
    alra_imputation::Bool=false,
    indents::Union{String, Nothing}=nothing,
    data_is_normalized::Bool=true
    ) 

    X = Matrix(df[:, 2:end]')
    cell_ids = df[:, 1]

    #---Check if the data path exists:
    if isnothing(data_path)

        projectpath = joinpath(@__DIR__, "../")
        data_path = projectpath * "data/tutorial/"
        if !isdir(data_path)
            mkdir(data_path)
            @info "No data path was provided. Creating a directory: $(data_path)"
        end

        @info "No data path was provided. Setting directory to: $(data_path)"
    end

    if isnothing(file_name)

        file_name = data_path * "Seurat_object.rds"

        @info "No file name was provided. Setting the file name to: Seurat_object.rds"
    end

    set_indents = false
    if !isnothing(indents) && indents in names(MD.obs_df)
        set_indents = true
    elseif !isnothing(indents)
        @error "The provided indent column does not exist in the metadata. Please choose an existing column."
    end

    if isnothing(assay)
        assay = "RNA"
        @info "No assay provided. Setting assay to RNA."
    end

    genenames = MD.featurename
    metadata = MD.obs_df

    R"library(Seurat)"

    @rput data_path
    @rput X
    @rput genenames
    @rput normalize_data
    @rput alra_imputation
    @rput metadata
    @rput set_indents
    @rput indents
    @rput assay
    @rput data_is_normalized
    @rput cell_ids

    # Create a Seurat object:
    R"""
    colnames(X) <- cell_ids
    seurat_obj <- CreateSeuratObject(counts = X, assay = assay)
    rownames(seurat_obj) <- genenames

    rownames(metadata) <- colnames(seurat_obj)
    seurat_obj <- AddMetaData(object = seurat_obj, metadata = metadata)
    """

    R"""
    if (set_indents) {
        Idents(seurat_obj) <- seurat_obj@meta.data[[indents]]
    }
    """

    R"""
    if (data_is_normalized) {
        LayerData(seurat_obj, assay = assay, layer = "data") <- LayerData(seurat_obj, assay = assay, layer = "counts")
    }
    """

    R"""
    #---Normalize the data:
    if (normalize_data) {
        print("Normalizing data ...")
        seurat_obj <- NormalizeData(seurat_obj)
    }
    """

    #---Impute the data:
    R"""
    if (alra_imputation) {
        print("Performing ALRA imputation ...")
        seurat_obj <- SeuratWrappers::RunALRA(seurat_obj, verbose = F)
    }
    """
    
    # Saving the Seurat object:
    R"""
    saveRDS(seurat_obj, file = paste0(data_path, "Seurat_object.rds"))
    """

    # Remove all objects from the R workspace:
    R"""
    rm(list=ls())
    gc()
    """
end

# Wrapper for the run NICHES function from (REF): Acts the same as the R function but currently, custom LR-databases ar not supported.
#' @param species string. The species of the object that is being processed. Only required when LR.database = 'fantom5' with species being 'human','mouse','rat', or 'pig', or LR.database = 'omnipath' with species being 'human','mouse', or 'rat'.
#' @param LR.database string. Default: "fantom5". Currently accepts "fantom5","omnipath", or "custom".
function run_NICHES_wrapper(file_path::String; 
    data_path::Union{String, Nothing}=nothing,
    normalize_data::Bool=false, 
    alra_imputation::Bool=false, 
    assay::String="alra", 
    species::String="rat", 
    LR_database::String="fantom5", 
    cell_types::String="cell_types", 
    min_cells_per_ident::Union{Int, Nothing}=nothing,
    min_cells_per_gene::Union{Int, Nothing}=nothing,
    meta_data_to_map::Union{AbstractVector{String}, Nothing}=nothing,
    position_x::Union{String, Nothing}=nothing,
    position_y::Union{String, Nothing}=nothing,
    CellToCell::Bool=true,
    CellToSystem::Bool=false,
    SystemToCell::Bool=false,
    CellToCellSpatial::Bool=false,
    CellToNeighborhood::Bool=false,
    NeighborhoodToCell::Bool=false,
    n_neighbors::Int=4,
    rad_set::Union{Int ,Nothing}=nothing,
    blend::String="mean",
    output_format::String="seurat"
    #custom_LR_database::Union{, Nothing}=nothing, #currently not supported ...
    )


    #---Load the required R packages:
    R"""
    library(Seurat)
    library(NICHES)
    library(SeuratWrappers)
    """

    #---Check if the data path exists:
    if isnothing(data_path)

        projectpath = joinpath(@__DIR__, "../")
        data_path = projectpath * "data/tutorial/"
        if !isdir(data_path)
            mkdir(data_path)
            println("No datapath was provided. Creating a directory: $(data_path)")
        end

        println("No datapath was provided. Setting directory to: $(data_path)")
    end

    #---Transfer the function variables to R:
    @rput file_path
    @rput alra_imputation
    @rput assay
    @rput species
    @rput LR_database
    @rput cell_types
    @rput data_path
    @rput normalize_data
    @rput position_x
    @rput position_y
    @rput min_cells_per_ident
    @rput min_cells_per_gene
    @rput meta_data_to_map
    @rput n_neighbors
    @rput rad_set
    @rput blend
    @rput output_format
    @rput CellToCell
    @rput CellToSystem
    @rput SystemToCell
    @rput CellToCellSpatial
    @rput CellToNeighborhood
    @rput NeighborhoodToCell
    
    #---Run the NICHES function:
    R"""
    seurat_obj <- readRDS(file_path)

    #---Normalize the data:
    if (normalize_data) {
        print("Normalizing data ...")
        seurat_obj <- NormalizeData(seurat_obj)
    }
    """

    #---Impute the data:
    R"""
    if (alra_imputation) {
        print("Performing ALRA imputation ...")
        seurat_obj <- SeuratWrappers::RunALRA(seurat_obj, verbose = F)
    }
    """

    #---Run NICHES:
    R"""
    print("Running NICHES ...")
    niches_obj <- RunNICHES(seurat_obj,
                        assay = assay, 
                        species = species,
                        LR.database = LR_database,
                        cell_types = cell_types,
                        meta.data.to.map = meta_data_to_map,
                        k = n_neighbors,
                        rad.set = rad_set,
                        position.x = position_x,
                        position.y = position_y,
                        blend = blend,
                        min.cells.per.ident = min_cells_per_ident,
                        min.cells.per.gene = min_cells_per_gene,
                        CellToCell = CellToCell,
                        CellToSystem = CellToSystem,
                        SystemToCell = SystemToCell,
                        CellToCellSpatial = CellToCellSpatial,
                        CellToNeighborhood = CellToNeighborhood,
                        NeighborhoodToCell = NeighborhoodToCell,
                        output_format = output_format
    )
    """

    #---Save the results:
    R"""
    print("Saving NICHES results ...")
    if (CellToCell) {
        niches_CtC <- niches_obj$CellToCell
        saveRDS(niches_CtC, file = paste0(data_path, "NICHES_CellToCell.rds"))

    }
    if (CellToSystem) {
        niches_CtS <- niches_obj$CellToSystem
        saveRDS(niches_CtS, file = paste0(data_path, "NICHES_CellToSystem.rds"))

    }
    if (SystemToCell) {
        niches_StC <- niches_obj$SystemToCell
        saveRDS(niches_StC, file = paste0(data_path, "NICHES_SystemToCell.rds"))

    }
    if (CellToCellSpatial) {
        niches_CtCspatial <- niches_obj$CellToCellSpatial
        saveRDS(niches_CtCspatial, file = paste0(data_path, "NICHES_CellToCell_Spatial.rds"))

    }
    if (CellToNeighborhood) {
        niches_CtN <- niches_obj$CellToNeighborhood
        saveRDS(niches_CtN, file = paste0(data_path, "NICHES_CellToNeighborhood.rds"))

    }
    if (NeighborhoodToCell) {
        niches_NtC <- niches_obj$NeighborhoodToCell
        saveRDS(niches_NtC, file = paste0(data_path, "NICHES_NeighborhoodToCell.rds"))
    } 

    #---Clean up:
    rm(list=ls())
    gc()
    """

end

#ToDo: Add versions for handling SystemToCell, CellToSystem, NeighborhoodToCell, and CellToNeighborhood.
function load_CCIM_CtC(file_path::String; min_obs::Union{Int, Nothing}=20, min_features::Union{Int, Nothing}=5)

    R"library(Seurat)"

    @rput file_path  
    
    @info "Loading the NICHES object from: $(file_path)"

    R"""
    NICHES_obj <- readRDS(file_path);
    X <- as.matrix(GetAssayData(object = NICHES_obj, assay = "CellToCell", slot = "counts"));
    SenderType <- NICHES_obj@meta.data$SendingType;
    ReceiverType <- NICHES_obj@meta.data$ReceivingType;
    interaction<- rownames(NICHES_obj);
    CellTypePair <- NICHES_obj@meta.data$VectorType;
    """

    #---Convert the data to Julia:
    @rget SenderType
    @rget ReceiverType
    @rget interaction
    @rget X
    @rget CellTypePair

    R"rm(list = ls())"
    R"gc()"

    SenderType = String.(SenderType);
    ReceiverType = String.(ReceiverType);
    interaction = String.(interaction);
    CellTypePair = String.(CellTypePair);

    X = Matrix{Float32}(transpose(X));

    #---Filter interactions and cell pairs:
    if !isnothing(min_features)
        @info "Filtering cell pairs ..."
        X, sel_cellpairs = filter_observations(X; min_features=min_features)

        CellTypePair = CellTypePair[sel_cellpairs]
        SenderType = SenderType[sel_cellpairs]
        ReceiverType = ReceiverType[sel_cellpairs]
    end

    if !isnothing(min_obs)
        @info "Filtering interactions ..."
        X, sel_interactions = filter_features(X; min_obs=min_obs)

        interaction = interaction[sel_interactions]

    end

    #---Scale the data:
    @info "Scaling the data ..."
    X_st = scale(X);

    #---Summarize meta data information:
    MD = MetaData(; featurename=interaction);
    MD.obs_df[!, :SenderType] = SenderType;
    MD.obs_df[!, :ReceiverType] = ReceiverType;
    MD.obs_df[!, :CellTypePair] = CellTypePair;

    n, p = size(X);

    @info "Finished process! The datamatrix has a sparsity level of $(round(sum(X .== 0)/length(X), digits=2) * 100)% and consists of $(n) observations, i.e., cell pairs and $(p) features, i.e., ligand-receptor interactions.";

    return X, X_st, MD
end

function load_CCIM_StC(file_path::String; min_obs::Union{Int, Nothing}=20, min_features::Union{Int, Nothing}=5)

    R"library(Seurat)"

    @rput file_path  
    
    @info "Loading the NICHES object from: $(file_path)"

    R"""
    NICHES_obj <- readRDS(file_path);
    X <- as.matrix(GetAssayData(object = NICHES_obj, assay = "SystemToCell", slot = "counts"));
    ReceiverType <- NICHES_obj@meta.data$ReceivingType;
    interaction<- rownames(NICHES_obj);
    """

    #---Convert the data to Julia:
    @rget ReceiverType
    @rget interaction
    @rget X

    R"rm(list = ls())"
    R"gc()"

    ReceiverType = String.(ReceiverType);
    interaction = String.(interaction);

    X = Matrix{Float32}(transpose(X));

    #---Filter interactions and cell pairs:
    if !isnothing(min_features)
        @info "Filtering cell pairs ..."
        X, sel_cellpairs = filter_observations(X; min_features=min_features)

        ReceiverType = ReceiverType[sel_cellpairs]
    end

    if !isnothing(min_obs)
        @info "Filtering interactions ..."
        X, sel_interactions = filter_features(X; min_obs=min_obs)

        interaction = interaction[sel_interactions]

    end

    #---Scale the data:
    @info "Scaling the data ..."
    X_st = scale(X);

    #---Summarize meta data information:
    MD = MetaData(; featurename=interaction);
    MD.obs_df[!, :ReceiverType] = ReceiverType;

    n, p = size(X);

    @info "Finished process! The datamatrix has a sparsity level of $(round(sum(X .== 0)/length(X), digits=2) * 100)% and consists of $(n) observations, i.e., cell pairs and $(p) features, i.e., ligand-receptor interactions.";

    return X, X_st, MD
end

function load_CCIM_CtS(file_path::String; min_obs::Union{Int, Nothing}=20, min_features::Union{Int, Nothing}=5)

    R"library(Seurat)"

    @rput file_path  
    
    @info "Loading the NICHES object from: $(file_path)"

    R"""
    NICHES_obj <- readRDS(file_path);
    X <- as.matrix(GetAssayData(object = NICHES_obj, assay = "CellToSystem", slot = "counts"));
    SenderType <- NICHES_obj@meta.data$SendingType;
    interaction<- rownames(NICHES_obj);
    """

    #---Convert the data to Julia:
    @rget SenderType
    @rget interaction
    @rget X

    R"rm(list = ls())"
    R"gc()"

    SenderType = String.(SenderType);
    interaction = String.(interaction);

    X = Matrix{Float32}(transpose(X));

    #---Filter interactions and cell pairs:
    if !isnothing(min_features)
        @info "Filtering cell pairs ..."
        X, sel_cellpairs = filter_observations(X; min_features=min_features)

        SenderType = SenderType[sel_cellpairs]
    end

    if !isnothing(min_obs)
        @info "Filtering interactions ..."
        X, sel_interactions = filter_features(X; min_obs=min_obs)

        interaction = interaction[sel_interactions]

    end

    #---Scale the data:
    @info "Scaling the data ..."
    X_st = scale(X);

    #---Summarize meta data information:
    MD = MetaData(; featurename=interaction);
    MD.obs_df[!, :SenderType] = SenderType;

    n, p = size(X);

    @info "Finished process! The datamatrix has a sparsity level of $(round(sum(X .== 0)/length(X), digits=2) * 100)% and consists of $(n) observations, i.e., cell pairs and $(p) features, i.e., ligand-receptor interactions.";

    return X, X_st, MD
end

function load_CCIM_CtC_Spatial(file_path::String; min_obs::Union{Int, Nothing}=20, min_features::Union{Int, Nothing}=5)

    R"library(Seurat)"

    @rput file_path  
    
    @info "Loading the NICHES object from: $(file_path)"

    R"""
    NICHES_obj <- readRDS(file_path);
    X <- as.matrix(GetAssayData(object = NICHES_obj, assay = "CellToCellSpatial", slot = "counts"));
    SenderType <- NICHES_obj@meta.data$SendingType;
    ReceiverType <- NICHES_obj@meta.data$ReceivingType;
    interaction<- rownames(NICHES_obj);
    CellTypePair <- NICHES_obj@meta.data$VectorType;
    x_coords <- NICHES_obj@meta.data$x;
    y_coords<- NICHES_obj@meta.data$y;
    """

    #---Convert the data to Julia:
    @rget SenderType
    @rget ReceiverType
    @rget interaction
    @rget X
    @rget CellTypePair
    @rget x_coords
    @rget y_coords

    R"rm(list = ls())"
    R"gc()"

    SenderType = String.(SenderType);
    ReceiverType = String.(ReceiverType);
    interaction = String.(interaction);
    CellTypePair = String.(CellTypePair);
    x_coords = number_vector = parse.(Int, strip.(x_coords))
    y_coords = number_vector = parse.(Int, strip.(y_coords))

    X = Matrix{Float32}(transpose(X));

    #---Filter interactions and cell pairs:
    if !isnothing(min_features)
        @info "Filtering cell pairs ..."
        X, sel_cellpairs = filter_observations(X; min_features=min_features)

        CellTypePair = CellTypePair[sel_cellpairs]
        SenderType = SenderType[sel_cellpairs]
        ReceiverType = ReceiverType[sel_cellpairs]
    end

    if !isnothing(min_obs)
        @info "Filtering interactions ..."
        X, sel_interactions = filter_features(X; min_obs=min_obs)

        interaction = interaction[sel_interactions]

    end

    #---Scale the data:
    @info "Scaling the data ..."
    X_st = scale(X);

    #---Summarize meta data information:
    MD = MetaData(; featurename=interaction);
    MD.obs_df[!, :SenderType] = SenderType;
    MD.obs_df[!, :ReceiverType] = ReceiverType;
    MD.obs_df[!, :CellTypePair] = CellTypePair;
    MD.obs_df[!, :x] = x_coords;
    MD.obs_df[!, :y] = y_coords;

    n, p = size(X);

    @info "Finished process! The datamatrix has a sparsity level of $(round(sum(X .== 0)/length(X), digits=2) * 100)% and consists of $(n) observations, i.e., cell pairs and $(p) features, i.e., ligand-receptor interactions.";

    return X, X_st, MD
end

function load_CCIM_NtC(file_path::String; min_obs::Union{Int, Nothing}=20, min_features::Union{Int, Nothing}=5)

    R"library(Seurat)"

    @rput file_path  
    
    @info "Loading the NICHES object from: $(file_path)"

    R"""
    NICHES_obj <- readRDS(file_path);
    X <- as.matrix(GetAssayData(object = NICHES_obj, assay = "NeighborhoodToCell", slot = "counts"));
    ReceiverType <- NICHES_obj@meta.data$ReceivingType;
    interaction<- rownames(NICHES_obj);
    x_coords <- NICHES_obj@meta.data$x;
    y_coords<- NICHES_obj@meta.data$y;
    """

    #---Convert the data to Julia:
    @rget ReceiverType
    @rget interaction
    @rget X
    @rget x_coords
    @rget y_coords

    R"rm(list = ls())"
    R"gc()"

    ReceiverType = String.(ReceiverType);
    interaction = String.(interaction);
    x_coords = number_vector = parse.(Int, strip.(x_coords))
    y_coords = number_vector = parse.(Int, strip.(y_coords))

    X = Matrix{Float32}(transpose(X));

    #---Filter interactions and cell pairs:
    if !isnothing(min_features)
        @info "Filtering cell pairs ..."
        X, sel_cellpairs = filter_observations(X; min_features=min_features)

        ReceiverType = ReceiverType[sel_cellpairs]
    end

    if !isnothing(min_obs)
        @info "Filtering interactions ..."
        X, sel_interactions = filter_features(X; min_obs=min_obs)

        interaction = interaction[sel_interactions]

    end

    #---Scale the data:
    @info "Scaling the data ..."
    X_st = scale(X);

    #---Summarize meta data information:
    MD = MetaData(; featurename=interaction);
    MD.obs_df[!, :ReceiverType] = ReceiverType;
    MD.obs_df[!, :x] = x_coords;
    MD.obs_df[!, :y] = y_coords;

    n, p = size(X);

    @info "Finished process! The datamatrix has a sparsity level of $(round(sum(X .== 0)/length(X), digits=2) * 100)% and consists of $(n) observations, i.e., cell pairs and $(p) features, i.e., ligand-receptor interactions.";

    return X, X_st, MD
end

function load_CCIM_CtN(file_path::String; min_obs::Union{Int, Nothing}=20, min_features::Union{Int, Nothing}=5)

    R"library(Seurat)"

    @rput file_path  
    
    @info "Loading the NICHES object from: $(file_path)"

    R"""
    NICHES_obj <- readRDS(file_path);
    X <- as.matrix(GetAssayData(object = NICHES_obj, assay = "CellToNeighborhood", slot = "counts"));
    SenderType <- NICHES_obj@meta.data$SendingType;
    interaction<- rownames(NICHES_obj);
    x_coords <- NICHES_obj@meta.data$x;
    y_coords<- NICHES_obj@meta.data$y;
    """

    #---Convert the data to Julia:
    @rget SenderType
    @rget interaction
    @rget X
    @rget x_coords
    @rget y_coords

    R"rm(list = ls())"
    R"gc()"

    SenderType = String.(SenderType);
    interaction = String.(interaction);
    x_coords = number_vector = parse.(Int, strip.(x_coords))
    y_coords = number_vector = parse.(Int, strip.(y_coords))

    X = Matrix{Float32}(transpose(X));

    #---Filter interactions and cell pairs:
    if !isnothing(min_features)
        @info "Filtering cell pairs ..."
        X, sel_cellpairs = filter_observations(X; min_features=min_features)

        SenderType = SenderType[sel_cellpairs]
    end

    if !isnothing(min_obs)
        @info "Filtering interactions ..."
        X, sel_interactions = filter_features(X; min_obs=min_obs)

        interaction = interaction[sel_interactions]

    end

    #---Scale the data:
    @info "Scaling the data ..."
    X_st = scale(X);

    #---Summarize meta data information:
    MD = MetaData(; featurename=interaction);
    MD.obs_df[!, :SenderType] = SenderType;
    MD.obs_df[!, :x] = x_coords;
    MD.obs_df[!, :y] = y_coords;

    n, p = size(X);

    @info "Finished process! The datamatrix has a sparsity level of $(round(sum(X .== 0)/length(X), digits=2) * 100)% and consists of $(n) observations, i.e., cell pairs and $(p) features, i.e., ligand-receptor interactions.";

    return X, X_st, MD
end