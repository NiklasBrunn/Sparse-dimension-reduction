#---Functions for result visualization:

function vegaheatmap(Z::AbstractMatrix; 
    path::String=joinpath(@__DIR__, "../") * "heatmap.pdf", 
    Title::String=" ",
    xlabel::String="Latent dimension", 
    ylabel::String="Observation",
    legend_title::String="Value",
    color_field::String="value:q",
    scheme::Union{Nothing,String}="blueorange",
    domain_mid::Union{Nothing, Int}=0,
    range::Union{Nothing, Vector{String}}=nothing,
    sortx::String="ascending",
    sorty::String="descending",
    Width::Int=800, 
    Height::Int=600,
    save_plot::Bool=false,
    axis_labelFontSize::AbstractFloat=14.0,
    axis_titleFontSize::AbstractFloat=14.0,
    legend_labelFontSize::AbstractFloat=12.5,
    legend_titleFontSize::AbstractFloat=14.0,
    legend_symbolSize::AbstractFloat=180.0,
    title_fontSize::AbstractFloat=16.0,
    legend_gradientThickness::AbstractFloat=20.0,
    legend_gradientLength::AbstractFloat=200.0,
    show_axis::Bool=true
    )

    n, p = size(Z)
    df = stack(DataFrame(Z', :auto), 1:n)
    df[!,:variable] = repeat(1:n, inner=p)
    df[!,:observation] = repeat(1:p, n)

    if isnothing(scheme) && isnothing(range)
        error("Please provide either a color scheme or a color range")
    end

    if !isnothing(scheme)
        if !isnothing(domain_mid)
            if show_axis==true
                vega_hmap = df |> @vlplot(mark=:rect, width=Width, height=Height, 
                                    title={text=Title, fontSize=title_fontSize}, 
                                    encoding={ 
                                        x={"observation:o", sort=sortx, axis={grid=false, title=xlabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}}, 
                                        y={"variable:o", sort=sorty, axis={grid=false, title=ylabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}},
                                        color={color_field, scale={scheme=scheme, domainMid=domain_mid}, label="Value", 
                                        legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                        gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}
                                    } 
                ) 
            else
                vega_hmap = df |> @vlplot(mark=:rect, width=Width, height=Height, 
                                    title={text=Title, fontSize=title_fontSize}, 
                                    encoding={ 
                                        x={"observation:o", sort=sortx, axis=nothing}, 
                                        y={"variable:o", sort=sorty, axis=nothing},
                                        color={color_field, scale={scheme=scheme, domainMid=domain_mid}, label="Value", 
                                        legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                        gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}
                                    } 
                ) 
            end
        else
            if show_axis==true
                vega_hmap = df |> @vlplot(mark=:rect, width=Width, height=Height, 
                                    title={text=Title, fontSize=title_fontSize}, 
                                    encoding={ 
                                        x={"observation:o", sort=sortx, axis={grid=false, title=xlabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}}, 
                                        y={"variable:o", sort=sorty, axis={grid=false, title=ylabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}},
                                        color={color_field, scale={scheme=scheme}, label="Value", 
                                        legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                        gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}
                                    } 
                ) 
            else
                vega_hmap = df |> @vlplot(mark=:rect, width=Width, height=Height, 
                                title={text=Title, fontSize=title_fontSize}, 
                                encoding={ 
                                    x={"observation:o", sort=sortx, axis=nothing}, 
                                    y={"variable:o", sort=sorty, axis=nothing},
                                    color={color_field, scale={scheme=scheme}, label="Value", 
                                    legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                    gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}
                                } 
                ) 
            end
        end
    else
        if !isnothing(domain_mid)
            if show_axis==true
                vega_hmap = df |> @vlplot(mark=:rect, width=Width, height=Height, 
                                    title={text=Title, fontSize=title_fontSize}, 
                                    encoding={ 
                                        x={"observation:o", sort=sortx, axis={grid=false, title=xlabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}}, 
                                        y={"variable:o", sort=sorty, axis={grid=false, title=ylabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}},
                                        color={color_field, scale={range=range, domainMid=domain_mid}, label="Value", 
                                        legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                        gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}
                                    } 
                ) 
            else
                vega_hmap = df |> @vlplot(mark=:rect, width=Width, height=Height, 
                                    title={text=Title, fontSize=title_fontSize}, 
                                    encoding={ 
                                        x={"observation:o", sort=sortx, axis=nothing}, 
                                        y={"variable:o", sort=sorty, axis=nothing},
                                        color={color_field, scale={range=range, domainMid=domain_mid}, label="Value", 
                                        legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                        gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}
                                    } 
                ) 
            end
        else
            if show_axis==true
                vega_hmap = df |> @vlplot(mark=:rect, width=Width, height=Height, 
                                    title={text=Title, fontSize=title_fontSize}, 
                                    encoding={ 
                                        x={"observation:o", sort=sortx, axis={grid=false, title=xlabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}}, 
                                        y={"variable:o", sort=sorty, axis={grid=false, title=ylabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}},
                                        color={color_field, scale={range=range}, label="Value", 
                                        legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                        gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}
                                    } 
                ) 
            else
                vega_hmap = df |> @vlplot(mark=:rect, width=Width, height=Height, 
                                    title={text=Title, fontSize=title_fontSize}, 
                                    encoding={ 
                                        x={"observation:o", sort=sortx, axis=nothing}, 
                                        y={"variable:o", sort=sorty, axis=nothing},
                                        color={color_field, scale={range=range}, label="Value", 
                                        legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                        gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}
                                    } 
                ) 
            end
        end
    end



    if save_plot == true
        vega_hmap |> VegaLite.save(path)
    end

    return vega_hmap
end

function vegascatterplot(embedding::AbstractMatrix, labels::AbstractVector; 
    path::String=joinpath(@__DIR__, "../") * "scatter.pdf",
    Title::String="",
    Width::Int=600, 
    Height::Int=600,
    legend_title::String="value",
    color_field::String="labels:o",
    scheme::Union{Nothing, String}="category20",
    domain_mid::Union{Nothing, Int}=0,
    range::Union{Nothing, Vector{String}}=nothing,
    save_plot::Bool=true,
    marker_size::String="15",
    legend_labelFontSize::AbstractFloat=24.0,
    legend_titleFontSize::AbstractFloat=28.0,
    legend_symbolSize::AbstractFloat=240.0,
    title_fontSize::AbstractFloat=28.0,
    legend_gradientThickness::AbstractFloat=20.0,
    legend_gradientLength::AbstractFloat=200.0
    )

    df = DataFrame(Embedding1 = embedding[:,1], Embedding2 = embedding[:,2], labels = labels) 

    if isnothing(scheme) && isnothing(range)
        error("Please provide either a color scheme or a color range")
    end

    if !isnothing(scheme)
        if !isnothing(domain_mid)
            umap_plot = df |> @vlplot(mark={"point", size=marker_size, filled="true"},
                            title={text=Title, fontSize=title_fontSize},
                            encoding={
                                x={:Embedding1, type=:quantitative, axis=nothing},   
                                y={:Embedding2, type=:quantitative, axis=nothing},    
                                color={color_field, scale={scheme=scheme, domainMid=domain_mid}, 
                                legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}   
                            },
                            config={view={stroke=nothing}},
                            width=Width, height=Height
            ) 
        else
            umap_plot = df |> @vlplot(mark={"point", size=marker_size, filled="true"},
                            title={text=Title, fontSize=title_fontSize},
                            encoding={
                                x={:Embedding1, type=:quantitative, axis=nothing},   
                                y={:Embedding2, type=:quantitative, axis=nothing},    
                                color={color_field, scale={scheme=scheme}, 
                                legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}   
                            },
                            config={view={stroke=nothing}},
                            width=Width, height=Height
            ) 
        end
    else
        if !isnothing(domain_mid)
            umap_plot = df |> @vlplot(mark={"point", size=marker_size, filled="true"},
                            title={text=Title, fontSize=title_fontSize},
                            encoding={
                                x={:Embedding1, type=:quantitative, axis=nothing},   
                                y={:Embedding2, type=:quantitative, axis=nothing},    
                                color={color_field, scale={range=range, domainMid=domain_mid}, 
                                legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}   
                            },
                            config={view={stroke=nothing}},
                            width=Width, height=Height
            ) 
        else
            umap_plot = df |> @vlplot(mark={"point", size=marker_size, filled="true"},
                            title={text=Title, fontSize=title_fontSize},
                            encoding={
                                x={:Embedding1, type=:quantitative, axis=nothing},   
                                y={:Embedding2, type=:quantitative, axis=nothing},    
                                color={color_field, scale={range=range}, 
                                legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}   
                            },
                            config={view={stroke=nothing}},
                            width=Width, height=Height
            ) 
        end
    end


    if save_plot == true
        umap_plot |> VegaLite.save(path)
    end

    return umap_plot
end

function create_colored_vegascatterplots(embedding::AbstractMatrix, Z::AbstractMatrix;
    path::String=joinpath(@__DIR__, "../"),
    filename::String="BAE_dim",
    filetype::String=".pdf",
    Title::String="",
    Width::Int=600, 
    Height::Int=600,
    legend_title::String="value",
    color_field::String="labels:o",
    scheme::Union{Nothing, String}="category20",
    domain_mid::Union{Nothing, Int}=0,
    range::Union{Nothing, Vector{String}}=nothing,
    save_plot::Bool=true,
    marker_size::String="15",
    legend_labelFontSize::AbstractFloat=24.0,
    legend_titleFontSize::AbstractFloat=28.0,
    legend_symbolSize::AbstractFloat=240.0,
    title_fontSize::AbstractFloat=28.0,
    legend_gradientThickness::AbstractFloat=20.0,
    legend_gradientLength::AbstractFloat=200.0
    )

    for k in 1:size(Z, 1)
        vegascatterplot(embedding, Z[k, :]; 
                        path=path*filename*"_$(k)_"*filetype, Title=Title,Width=Width, Height=Height, legend_title=legend_title,
                        color_field=color_field, scheme=scheme, domain_mid=domain_mid, range=range,
                        save_plot=save_plot, marker_size=marker_size, legend_labelFontSize=legend_labelFontSize,
                        legend_titleFontSize=legend_titleFontSize, legend_symbolSize=legend_symbolSize,
                        title_fontSize=title_fontSize, legend_gradientThickness=legend_gradientThickness,
                        legend_gradientLength=legend_gradientLength
        )
    end
end

function TopFeaturesPerCluster_scatterplot(df::DataFrame, key::String; 
    top_n::Int=10, 
    size::Tuple=(800, 700)
    )

    if length(df.Scores) < top_n
        top_n = length(df.Scores)
    end

    values = df.Scores
    selected_labels = df.Features

    # Normalize by the maximum absolute value
    normalized_values = (values .- minimum(values)) ./ (maximum(values) - minimum(values)) #values / maximum(values)
    
    # Select the top_n values and their corresponding labels
    top_values = normalized_values[1:top_n]
    top_labels = selected_labels[1:top_n]

    # Compute colors based on the blueorange scheme
    color_map(val) = val < 0 ? ColorSchemes.vik[0.5 * (1 + val)] : ColorSchemes.vik[0.5 + 0.5 * val]
    colors = color_map.(top_values)

    # Determine y-axis limits based on data range
    y_lower_limit = -0.1 
    y_upper_limit = max(1, maximum(top_values) + 0.1)  

    
    # Plot scatter plot with custom x-axis labels and colors
    p = scatter(1:top_n, top_values, 
                size=size,
                ylabel="Max.-norm. coefficient", 
                guidefontsize=14,
                title="Top $(top_n) features for cluster" * key * " ($(length(df.Scores)) nonzero features)",
                titlefontsize=14,
                legend=false, 
                markersize=8,
                xticks=(1:top_n, top_labels),
                tickfontsize=14,
                ylims=(y_lower_limit, y_upper_limit),
                color=colors,
                grid=false,
                xrotation = 90)
    
    hline!(p, [0], color=:black, linewidth=1.5, linestyle=:dash) 
    return p
end

function normalizedFeatures_scatterplot(vec::AbstractVector, labels::AbstractVector, dim::Int; 
    top_n::Int=15, 
    size::Tuple=(700, 600)
    )

    # Filter out zeros and get indices of nonzero elements
    non_zero_indices = findall(x -> x != 0, vec)

    if length(non_zero_indices) < top_n
        top_n = length(non_zero_indices)
    end

    non_zero_values = vec[non_zero_indices]
    selected_labels = labels[non_zero_indices]

    # Normalize by the maximum absolute value
    normalized_values = non_zero_values / maximum(abs.(non_zero_values))

    # Get indices that would sort the normalized_values by absolute magnitude (in descending order)
    sorted_indices = sortperm(abs.(normalized_values), rev=true)
    
    # Select the top_n values and their corresponding labels
    top_values = normalized_values[sorted_indices[1:top_n]]
    top_labels = selected_labels[sorted_indices[1:top_n]]

    # Compute colors based on the blueorange scheme
    color_map(val) = val < 0 ? ColorSchemes.vik[0.5 * (1 + val)] : ColorSchemes.vik[0.5 + 0.5 * val]
    colors = color_map.(top_values)

    # Determine y-axis limits based on data range
    y_lower_limit = min(-1, minimum(top_values) - 0.1) 
    y_upper_limit = max(1, maximum(top_values) + 0.1)  

    
    # Plot scatter plot with custom x-axis labels and colors
    p = scatter(1:top_n, top_values, 
                size=size,
                ylabel="Max.-norm. coefficient", 
                guidefontsize=14,
                title="Top $(top_n) features in latent dimension $(dim) ($(length(non_zero_indices)) nonzero features)",
                titlefontsize=14,
                legend=false, 
                markersize=8,
                xticks=(1:top_n, top_labels),
                tickfontsize=14,
                ylims=(y_lower_limit, y_upper_limit),
                color=colors,
                grid=false,
                xrotation = 90)
    
    hline!(p, [0], color=:black, linewidth=1.5, linestyle=:dash) 
    return p
end

function FeaturePlots(dict::Dict, featurenames::AbstractVector{String}, X::AbstractMatrix, embedding::AbstractMatrix; 
    top_n::Int=10,
    marker_size::String="15", 
    Width::Int=800, 
    Height::Int=800, 
    fig_type::String=".pdf",
    path::String=joinpath(@__DIR__, "../"),
    legend_title::String="log1p",
    color_field::String="labels:q",
    scheme::Union{Nothing, String}="lightgreyred", #"orangered"
    domain_mid::Union{Nothing, Int}=nothing,
    range::Union{Nothing, Vector{String}}=nothing
    )

    for ldim in 1:length(dict)

        k = minimum([top_n, length(dict["$(ldim)"].Features)])

        if k > 0

            figurespath_ldim = path * "/TopFeaturesCluster_$(ldim)"
            if !isdir(figurespath_ldim)
                # Create the folder if it does not exist
                mkdir(figurespath_ldim)
            end

            sel_features = dict["$(ldim)"].Features[1:k]; #top_n
            for feature in 1:k  #top_n
                featurename = sel_features[feature]
                sel_ind = findall(x->x==featurename, featurenames);
                vegascatterplot(embedding, vec(X[:, sel_ind]); 
                                path=figurespath_ldim * "/TopFeature_$(feature)_"* featurename *"_Topic_$(ldim)"*fig_type,
                                Title=Title="Top feature $(feature): " * featurename, Width=Width, Height=Height,
                                legend_title=legend_title, color_field=color_field,
                                scheme=scheme, domain_mid=domain_mid, range=range, save_plot=true,
                                marker_size=marker_size
                )
            end
            
        end

    end
end

function origData_FeaturePlots(X::AbstractMatrix{T}, embedding::AbstractMatrix{T}, MD::MetaData, genenames::AbstractVector{String}; 
    n_features::Int=4,
    marker_size::String="50",
    color_range::Union{AbstractVector{String}, Nothing}=nothing,
    color_scheme::Union{String, Nothing}="reds",
    legend_title::String="Counts",
    Width::Int=800,
    Height::Int=800,
    file_type::String=".png",
    figures_path::Union{String, Nothing}=nothing
    ) where T

    if isnothing(MD.Top_features)
        @error "No top features found in metadata!"
    end

    if isnothing(figures_path)

        figures_path = joinpath(@__DIR__, "../figures/tutorial/origData_FeaturePlots")
        @info "No figures path provided. Figures will be saved to: $figures_path."

        if !isdir(figures_path)
            mkdir(figures_path)
        end
    else
        figures_path = joinpath(figures_path, "origData_FeaturePlots")

        if !isdir(figures_path)
            mkdir(figures_path)
        end

    end

    df_origData = DataFrame(X, :auto);
    rename!(df_origData, Symbol.(genenames)) 

    for cluster in 1:length(MD.Top_features)
        n_features = 4;
        topinteractions = MD.Top_features["$(cluster)"][:, :Features]

        if length(topinteractions) > 0

            if length(topinteractions) < n_features
                n_features = length(topinteractions)
            end

            split_identifyer = '\u2014'
            topinteractions = vcat(split.(topinteractions[1:n_features], split_identifyer)...)

            if !isdir(figures_path * "/Cluster_$(cluster)")
                mkdir(figures_path * "/Cluster_$(cluster)")
            end

            for gene in topinteractions
                vegascatterplot(embedding, df_origData[!, Symbol(gene)]; 
                   path=figures_path * "/Cluster_$(cluster)/" * "Gene_$(gene)_origData" * file_type,
                   Title="$(gene)", Width=Width, Height=Height,
                   legend_title=legend_title, color_field="labels:q",
                   scheme=color_scheme, domain_mid=nothing, range=color_range, save_plot=true,
                   marker_size=marker_size
                )
            end
        end

    end

end

function create_spatialInteractionPlots(file_path::String, MD::MetaData, X::AbstractMatrix{T}; 
    figures_path::Union{String, Nothing}=nothing,
    background_color::String="light", 
    legend_title::String="Value", 
    n_Interactions::Int=5, 
    marker_size::String="150", 
    fig_type::String=".png"
    ) where T

    if !isfile(file_path)
        @error "File not found!"
    end

    if MD.Top_features == nothing
        @error "No top features, i.e., interactions found in metadata!"
    end

    if !isnothing(figures_path) && !isdir(figures_path)
        @error "Directory not found!"

    elseif isnothing(figures_path)
        projectpath = joinpath(@__DIR__, "../")
        figures_path = projectpath * "/figures/"

        if !isdir(figures_path)
            mkdir(figures_path)
        end

        @info "No figurespath provided. Figures will be saved to the default directory: $(figures_path)"

    end

    color_range = []
    if background_color == "light"
        color_range = [
            "#000000", "#220022", "#440044", "#660066", "#880088", "#aa00aa", "#cc00cc", "#ee00ee",
            "#ff00ff", "#ff19ff", "#ff33ff", "#ff4cff", "#ff66ff", "#ff7fff", "#ff99ff", "#ffb2ff",
            "#ffccff", "#ffe5ff", "#ffccf5", "#ff99eb", "#ff66e0"
        ]

    elseif background_color == "dark"
        color_range = [
            "#fff5f5", "#ffe0e0", "#ffcccc", "#ffb8b8", "#ffa3a3", "#ff8f8f", "#ff7a7a", "#ff6666",
            "#ff5252", "#ff3d3d", "#ff2929", "#ff1414", "#ff0000", "#e50000", "#cc0000", "#b20000",
            "#990000", "#7f0000", "#660000", "#4c0000", "#330000"
        ]

    else
        @error "Background color not recognized!"
    end


    R"library(Seurat)"

    @rput file_path  
        
    @info "Loading the NICHES object from: $(file_path)"

    R"""
    NICHES_obj <- readRDS(file_path);
    x_coords <- NICHES_obj@meta.data$x
    y_coords<- NICHES_obj@meta.data$y
    """

    #---Convert the data to Julia:
    @rget x_coords
    @rget y_coords

    R"rm(list = ls())"
    R"gc()"

    x_coords = number_vector = parse.(Int, strip.(x_coords))
    y_coords = number_vector = parse.(Int, strip.(y_coords))
    cell_locations_CCIM = Matrix{Float32}(hcat(x_coords, y_coords))


    #---Create scatter plots of the UMAP embedding of the original data representation colored by expression levels of top selected genes for different clusters:
    if !isdir(figures_path * "/SpatialInteractionPlots")
        # Create the folder if it does not exist
        mkdir(figures_path * "/SpatialInteractionPlots")
    end

    FeaturePlots(MD.Top_features, MD.featurename, X, cell_locations_CCIM; 
        top_n=n_Interactions,
        marker_size=marker_size, 
        fig_type=fig_type,
        path=figures_path * "/SpatialInteractionPlots/",
        legend_title=legend_title,
        color_field="labels:q",
        scheme=nothing, 
        domain_mid=nothing,
        range=color_range
    )

    @info "Figures saved to: $(figures_path)!"
end

function track_coefficients(coefficients, dim; iters::Union{Int, Nothing}=nothing, xscale::Symbol=:log10)
    # Number of iterations and number of coefficients
    num_iterations = length(coefficients)
    num_coefficients = size(coefficients[1], 1)

    # Preallocate the coefficient matrix
    coeffs = zeros(Float32, num_coefficients, num_iterations)

    # Populate the coefficient matrix
    for iter in 1:num_iterations
        coeffs[:, iter] = coefficients[iter][:, dim]
    end

    # Handle the number of iterations to plot
    if typeof(iters) == Int && iters > num_iterations
        @warn "Number of iterations to plot is greater than the number of iterations in the data. Plotting all iterations."
        iters = num_iterations
    elseif isnothing(iters)
        iters = num_iterations
    end

    # Prepare data for plotting
    x = 1:iters
    y = coeffs[:, 1:iters]

    # Create the plot
    x_scale = string(xscale)
    pl = plot(x, y', xlabel="Iteration " * x_scale, ylabel="Coefficient value", title="Evolution of coefficients across training epochs", lw=2, legend=false, xscale=xscale)

    return pl
end

function plot_row_boxplots(Z; legend=false, xlabel="Row Index", ylabel="Value", palette=:tab20, saveplot=false, path=nothing)

    d, n = size(Z)
    
    # Create a vector to hold the positions for the boxplots
    positions = repeat(1:d, inner=n)
    
    # Create a vector to hold the values for the boxplots
    values = vec(Z')
    
    # Create a vector to hold the groupings (row index for each value)
    groupings = repeat(1:d, inner=n)
    
    # Create a DataFrame for plotting
    df = DataFrame(Row=positions, Value=values, Group=groupings)
    
    # Plot the boxplots with different colors
    b_plot = boxplot(df.Row, df.Value, group=df.Group, legend=legend, xlabel=xlabel, ylabel=ylabel, xticks=1:d, palette=palette)

    if saveplot
        savefig(b_plot, path)
    end
    
    display(b_plot)
end