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

0. **Download the repository**

0.1. **Open your terminal**
   - On macOS or Linux, open the Terminal application.
   - On Windows, you can use Command Prompt, PowerShell, or Git Bash.

0.2. **Navigate to your desired directory**
   - Use the `cd` command to change to the directory where you want to clone the repository.
   - Example (macOS): To change to a directory named `MyProjects` on your desktop, you would use:
     ```bash
     cd ~/Desktop/MyProjects
     ```
   - Example (Windows): To change to a directory named `MyProjects` on your desktop, you would use:
     ```bash
     cd C:\Users\[YourUsername]\Desktop\MyProjects
     ```
     
0.3. **Clone the repository**
   - Use the `git clone` command followed by the URL of the repository.
   - You can find the URL on the repository's GitHub page.
   - Example:
     ```bash
     git clone https://github.com/NiklasBrunn/BoostingAutoencoder/tree/main
     ```

1. **Install Julia**
   - To run the Julia scripts, [Julia v1.9.3](#https://julialang.org/downloads/oldreleases/) has to be downloaded and installed manually by the user. The required packages and their versions are specified in the `Project.toml` and `Manifest.toml` files in the main folder and automatically loaded/installed at the beginning of each script with the `Pkg.activate()` and `Pkg.instantiate()` commands. See [here](https://pkgdocs.julialang.org/v1.2/environments/) for more information on Julia environments. 

2. **Install R**
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
