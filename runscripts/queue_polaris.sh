NUM_NODES=2
qsub -A APSDataAnalysis -q debug -l select=${NUM_NODES}:system=polaris -l walltime=1:00:00 -l filesystems=home:eagle -l place=scatter /eagle/projects/APSDataAnalysis/mprince/lau/laue-parallel/runscripts/run_polaris.sh