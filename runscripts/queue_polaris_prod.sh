NUM_NODES=16
qsub -A APSDataAnalysis -q prod -l select=${NUM_NODES}:system=polaris -l walltime=3:00:00 -l filesystems=home:eagle -l place=scatter /eagle/projects/APSDataAnalysis/mprince/lau/laue-parallel/runscripts/run_polaris.sh