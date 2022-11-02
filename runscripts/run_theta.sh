cd /eagle/APSDataAnalysis/mprince/lau/laue-parallel

CONFIGPATH=/eagle/projects/APSDataAnalysis/mprince/lau/laue-parallel/configs/config-64.yml

echo ${CONFIGPATH}

aprun -n 64 -N 64 -d 1 -j 1 -cc depth \
    /eagle/APSDataAnalysis/mprince/lau/env_theta/lau_theta/bin/python \
    /eagle/projects/APSDataAnalysis/mprince/lau/laue-parallel/laue_parallel.py \
    ${CONFIGPATH} \
    --log_time \
    --h5_backup \
    --disable_recon

echo "Completed processing..."

aprun -n 64 -N 64 -d 1 -j 1 -cc depth \
    /eagle/APSDataAnalysis/mprince/lau/env_theta/lau_theta/bin/python \
    /eagle/projects/APSDataAnalysis/mprince/lau/laue-parallel/recon_parallel.py \
    ${CONFIGPATH} \