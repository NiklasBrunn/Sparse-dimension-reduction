module BoostingAutoEncoder

#check which ones are required ...
using Flux;
using Random; 
using Statistics;
using DelimitedFiles;
using Plots;
using LinearAlgebra;
using DataFrames;
using VegaLite;
using UMAP;
using ProgressMeter;
using ColorSchemes;
using CSV;
#using Distributions;
#using ExcelFiles;
#using XLSX;

include("BAE_architecture.jl");
include("utils.jl");
include("BAE_optimization.jl");
include("visualization.jl");
include("simulation.jl");

export 
    # BAE model architecture:
    Hyperparameter, MetaData, BoostingAutoencoder, generate_BAEdecoder,
    # BAE model optimization functions:
    calcunibeta, compL2Boost!, disentangled_compL2Boost!, train_BAE!,
    # plotting functions:
    vegaheatmap, vegascatterplot, create_colored_vegascatterplots, TopFeaturesPerCluster_scatterplot, normalizedFeatures_scatterplot, FeaturePlots,
    # data simulation functions:
    addstages!, sim_scRNAseqData,
    # utility functions:
    scale, get_latentRepresentation, generate_umap, find_zero_columns, split_vectors, slit_softmax, topFeatures_per_Cluster, hsl_to_hex
#

end