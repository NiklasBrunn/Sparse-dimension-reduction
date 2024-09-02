# The-role-of-sparse-dimension-reduction-in-the-analysis-of-single-cell-resolved-interactions

### Tabel of contents:

- [Installation](#Installation)
- [Launch Jupyter notebooks](#Launch-Jupyter-notebooks)


## Installation
- Git should be installed on your computer. You can download and install it from [Git's official website](https://git-scm.com/downloads).

0. **Clone the repository**:
  - You may want to navigate to a specific directory before cloning the repository.
  - ```bash
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
- Part of the presented workflow is the interaction with an R tool called `NICHES` via Julia. R code can be executed in Julia using the package `RCall` but this requires [R v.4.3.2](https://www.r-project.org) to be downloaded and installed on your computer. 
- The following R packages are required to run the embedded R code in the Julia scripts: `Seurat`, `SeuratData`, `SeuratWrappers`, `SeuratWrappers`
 - The R packages can be installed by running the Jupyter notebook called `install_R_dependencies.ipynb`.
 - Alternatively, open the terminal start R and install the required packages:
- ```bash
  R
  ```
- ```r
  install.packages(c("Seurat", "SeuratData", "SeuratWrapper"))
  library(devtools)
  install_github('msraredon/NICHES', ref = 'master')
  ```


## Launch Jupyter notebooks
1. **From terminal**
  - In the terminal run the following command to start a Jupyter notebook server with the Julia programming environment:
    ```bash
    julia -e 'using IJulia; notebook()'
    ```

2. **Using VS code**
  - ToDo: ...



