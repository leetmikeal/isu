set -eu

LABELS=(
alcon01
alcon06
alcon07
alcon08
alcon12
)

# MASTER=work/label
#TARGET=work/output_predict_190924
# TARGET=work/output_2d/temp2d

for L in ${LABELS[@]}; do
    echo ${L}
    isu predict \
        --in-dir work/input/${L} \
        --in-model work/model3d.h5 \
        --out-dir work/temp/3d/${L} \
        --verbose
done