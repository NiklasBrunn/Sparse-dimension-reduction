module BoostingAutoEncoder

#--- Used packages ---#
# Numerical and statistical packages
using Flux, Random, Statistics, LinearAlgebra, Distances

# Data handling packages
using DataFrames, CSV, DelimitedFiles

# Plotting and visualization packages
using Plots, VegaLite, ColorSchemes

# Dimensionality reduction and clustering packages
using UMAP, Clustering

# Utility packages
using ProgressMeter, RCall


#--- Included files ---#
include("BAE_architecture.jl");
include("utils.jl");
include("BAE_optimization.jl");
include("visualization.jl");
include("simulation.jl");


#--- Exported functions ---#
export 
    # BAE model architecture:
    Hyperparameters, MetaData, BoostingAutoencoder, generate_BAEdecoder,

    # BAE model optimization functions:
    get_unibeta, compL2Boost!, disentangled_compL2Boost!, train_BAE!, 

    # plotting functions:
    vegaheatmap, vegascatterplot, create_colored_vegascatterplots, TopFeaturesPerCluster_scatterplot, 
    normalizedFeatures_scatterplot, FeaturePlots, origData_FeaturePlots, create_spatialInteractionPlots, track_coefficients, 
    plot_row_boxplots,

    # data simulation functions:
    addstages!, sim_scRNAseqData,

    # utility functions:
    scale, NaNstoZeros!, Normalize, MinMaxNormalize, log1p, log1pNormalize, filter_observations, 
    filter_features, filter_cells_by_mitochondrial_content, get_latentRepresentation, generate_umap, 
    find_zero_columns, split_vectors, slit_softmax, topFeatures_per_Cluster, get_important_genes, hsl_to_hex, 
    load_rat_scRNAseq_data, load_spatial_mousebrain_data, create_seurat_object, run_NICHES_wrapper, load_CCIM_CtC, load_CCIM_StC, 
    load_CCIM_CtS, load_CCIM_CtC_Spatial, load_CCIM_NtC, load_CCIM_CtN
#

end