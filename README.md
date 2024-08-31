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