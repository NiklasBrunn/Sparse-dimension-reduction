# The-role-of-sparse-dimension-reduction-in-the-analysis-of-single-cell-resolved-interactions

## Required R Packages

The following R packages are required to run the embedded R code in the Julia scripts:

- `Seurat`
- `SeuratData`
- `SeuratWrappers`
- `NICHES`

You can install these packages in R by running:

```r
install.packages(c("Seurat", "SeuratData", "SeuratWrapper"))
library(devtools)
install_github('msraredon/NICHES', ref = 'master')
```

Or via Julia by running:

> using RCall;
  R"""
  install.packages(c("Seurat", "SeuratData", "SeuratWrapper"))
  library(devtools)
  install_github('msraredon/NICHES', ref = 'master')
  """

## Installation
- Git should be installed on your computer. You can download and install it from [Git's official website](https://git-scm.com/downloads).

0. **Clone the repository**:
  You may want to navigate to a specific directory before cloning the repository.
   ```bash
   git clone https://github.com/NiklasBrunn/Sparse-dimension-reduction

1. **Install Julia**
  - To run the Julia scripts, [Julia v1.9.3](https://julialang.org/downloads/oldreleases/) has to be downloaded and installed manually by the user. The required packages and their versions are specified in the `Project.toml` and `Manifest.toml` files in the main folder and automatically loaded/installed at the beginning of each notebook with the `Pkg.activate()` and `Pkg.instantiate()` commands. See [here](https://pkgdocs.julialang.org/v1.2/environments/) for more information on Julia environments. 

2. **Install the Julia Kernel for Jupyter**
  - To use Julia as a kernel in Jupyter notebooks, the IJulia package in Julia has to be installed:

  - 2.1 In the terminal (macOS) or command prompt or power shell (Windows) run:
   
  - ```bash
    julia
    ```

- 2.2 Then, in the Julia REPL, enter the following commands:

-   ```julia
    using Pkg
    Pkg.add("IJulia")
    ```


3. **Install R**
   - ...
   - To run the Python scripts, we included details about a [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment in (`environment.yml`) consisting of information about the Python version and used packages. A new conda environment can be created from this file. See [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) for more details about managing and creating conda environments. Follow these steps to set up your development environment:

2.1. **Navigate to the project directory**
   - Navigate to the directory of the cloned GitHub repository (macOS):
     ```bash
     cd ~/BoostingAutoencoder
     ```
   - (Windows):
     ```bash
     cd \BoostingAutoencoder
     ```
       
2.2. **Create the conda environment**
   - Create a new conda environment that is named as specified in the `environment.yml` file (in this case it is named `BAE-env`):
     ```bash
     conda env create -f environment.yml
     ```

2.3. **Use the BAE conda environment for running python code**
   - Once the environment is created, select it as the kernel for running the python code in the repository.


## Launch Jupyter notebooks
1. **From terminal**
  In the terminal run:

  - In the terminal run:
    ```bash
    julia -e 'using IJulia; notebook()'
    ```

2. **Using VS code**
  ...



