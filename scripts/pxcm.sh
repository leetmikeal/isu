set -eu

LABELS=(
alcon01
alcon06
alcon07
alcon08
alcon12
)

MASTER=work/label
#TARGET=work/output_predict_190924
#TARGET=work/output_2d/temp2d
#TARGET=work/temp/ensemble
TARGET=work/output
#TARGET=work/temp/2d
# TARGET=work/temp/3d

for L in ${LABELS[@]}; do
    echo ${L}
    isu analyze pxcm \
        --in-dir1 ${MASTER}/${L} \
        --in-dir2 ${TARGET}/${L} \
        --out-csv ${TARGET}/pxcm_${L}.csv \
        --verbose
done
exit