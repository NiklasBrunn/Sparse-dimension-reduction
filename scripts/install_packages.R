# install_packages.R
packages <- c("Seurat", "SeuratData", "SeuratWrapper", "NICHES")

install.packages(setdiff(packages, installed.packages()[,"Package"]), dependencies = TRUE)


###Julia Script to Check and Install R Packages: (add this in the notebooks and julia scripts!)
#using RCall

# Define the required R packages
#required_r_packages = ["ggplot2", "dplyr", "tidyr", "data.table"]

# Function to check and install missing R packages
#for pkg in required_r_packages
#    @rput pkg
#    R"""
#    if (!require(pkg, character.only = TRUE)) {
#        install.packages(pkg, dependencies = TRUE)
#    }
#    """
#end
#"""