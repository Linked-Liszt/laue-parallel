cd /eagle/APSDataAnalysis/mprince/lau/dev/laue-parallel/logs

CONFIGPATH=/eagle/projects/APSDataAnalysis/mprince/lau/dev/laue-parallel/configs/AL30/config-64.yml

echo ${CONFIGPATH}

aprun -n 64 -N 64 -d 4 -j 4 -cc depth \
    /eagle/APSDataAnalysis/mprince/lau/env_theta/lau_theta/bin/python \
    /eagle/projects/APSDataAnalysis/mprince/lau/dev/laue-parallel/laue_parallel.py \
    ${CONFIGPATH} \
    --profile \

echo "Completed processing..."

aprun -n 64 -N 64 -d 4 -j 4 -cc depth \
    /eagle/APSDataAnalysis/mprince/lau/env_theta/lau_theta/bin/python \
    /eagle/projects/APSDataAnalysis/mprince/lau/dev/laue-parallel/recon_parallel.py \
    ${CONFIGPATH} \