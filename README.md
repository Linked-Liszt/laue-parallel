# laue-holding

Scripts and surrounding infrastructure to run cold in parallel via spatial decomposition on the Theta and Polaris supercomputers. 

## Repository Organization

* `laue_parallel.py` performs the parallel computation, and `recon_parallel.py` reconstructs the individual process outputs from the first script
* These scripts are controlled by command-line arguments and by config files located in `configs`
* `analysis` contains various scripts that were useful for debugging the output of output files
* `runscripts` contains shell scripts with HPC settings embedded to queue debug and individual production runs
* `prod_scripts` contains python scripts to generate and run batches of jobs needed for full production runs