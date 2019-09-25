set -eu

LABELS=(
alcon01
alcon06
alcon07
alcon08
alcon12
)

for L in ${LABELS[@]}; do
    echo ${L}
    isu predict-2d \
        --in-settings setting.ini \
        --dataset ${L} \
        --verbose
    isu predict-3d \
        --in-settings setting.ini \
        --dataset ${L} \
        --verbose
    isu analyze ensemble \
        --in-dir1 work/temp/2d/${L} \
        --in-dir2 work/temp/3d/${L} \
        --out-dir work/temp/ensemble/${L} \
        --verbose
    isu analyze connection \
        --in-dir work/temp/ensemble/${L} \
        --out-dir work/output/${L} \
        --in-filename *.tif \
        --stat work/output/${L}_stat.log \
        --verbose
done