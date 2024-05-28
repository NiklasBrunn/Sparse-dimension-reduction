#---utils:
function scale(X::AbstractArray{T}; corrected_std::Bool=true, dims::Int=1) where T
    return (X .- mean(X, dims=dims)) ./ std(X, corrected=corrected_std, dims=dims)
end

function get_latentRepresentation(BAE::BoostingAutoencoder, X::AbstractMatrix{T}) where T
    Z = transpose(BAE.coeffs) * X
    return Z
end

function generate_umap(X::AbstractMatrix{T}, plotseed::Int; n_neighbors::Int=30, min_dist::Float64=0.4) where T
    Random.seed!(plotseed)  
    embedding = umap(X', n_neighbors=n_neighbors, min_dist=min_dist)'
    return embedding
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

function topFeatures_per_Cluster(BAE::BoostingAutoencoder, MD::MetaData; save_data::Bool=true, path::Union{Nothing, String}=nothing)

    TopFeatures_Cluster = Dict{String, DataFrame}();
    counter_var = 1;
    for l in 1:size(BAE.coeffs, 2)
        pos_inds = findall(x->x>0, BAE.coeffs[:, l])
        df_pos = DataFrame(Features=MD.featurename[pos_inds])
        df_pos[!, "Scores"] = BAE.coeffs[pos_inds, l]
        df_pos[!, "normScores"] = df_pos[!, "Scores"] ./ maximum(df_pos[!, "Scores"])
        sort!(df_pos, :Scores, rev=true)
        TopFeatures_Cluster["$(counter_var)"] = df_pos
        
        counter_var+=1
        neg_inds = findall(x->x>0, -BAE.coeffs[:, l])
        df_neg = DataFrame(Features=MD.featurename[neg_inds])
        df_neg[!, "Scores"] = -BAE.coeffs[neg_inds, l]
        df_neg[!, "normScores"] = df_neg[!, "Scores"] ./ maximum(df_neg[!, "Scores"])
        sort!(df_neg, :Scores, rev=true)
        TopFeatures_Cluster["$(counter_var)"] = df_neg

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