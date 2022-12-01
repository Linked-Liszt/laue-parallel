qsub -n 1 -t 60 -A APSDataAnalysis -q debug-cache-quad --attrs filesystems=home,eagle \
    /eagle/APSDataAnalysis/mprince/lau/dev/laue-parallel/runscripts/al/run_theta.sh
