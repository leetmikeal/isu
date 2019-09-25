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
        --in-settings setting.ini \
        --dataset ${L} \
        --verbose
    isu analyze connection \
        --in-settings setting.ini \
        --dataset ${L} \
        --verbose
        
done