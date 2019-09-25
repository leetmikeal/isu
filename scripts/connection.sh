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
    isu analyze connection \
        --in-dir work/temp/ensemble/${L} \
        --out-dir work/output/${L} \
        --in-filename *.tif \
        --stat work/output/${L}_stat.log \
        --verbose
done