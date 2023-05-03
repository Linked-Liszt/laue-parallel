# laue-parallel

Scripts and surrounding infrastructure to run cold in parallel via spatial decomposition on the Polaris supercomputer. 

## Repository Organization

* `laue_parallel.py` performs the parallel computation, and `recon_parallel.py` reconstructs the individual process outputs from the first script
* These scripts are controlled by command-line arguments described in the scripts and by the cold config files located in `configs`
* `analysis` contains various scripts that are useful for debugging output and logging files
* `runscripts` contains shell scripts with HPC settings embedded to queue debug and individual production runs
* `prod_scripts` contains python scripts to generate and run batches of jobs needed for full production runs


## Environment Setup

Pip installable requirements can be found in `requirements.txt`. These are additional requirements added on top of the [base anaconda environments](https://www.alcf.anl.gov/support/user-guides/polaris/data-science-workflows/python/index.html) provided by ALCF. Cloning the environment is recommended before attempting to add or modify packages. Additional external requirements and github links are described `requirements.txt` commented lines. 
