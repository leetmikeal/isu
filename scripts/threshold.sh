set -eu

LABELS=(
alcon01
alcon06
alcon07
alcon08
alcon12
)

INPUT=work/output
OUTPUT=work/output_threshold

for L in ${LABELS[@]}; do
    echo ${L}
    isu analyze threshold \
        --in-dir ${INPUT}/${L} \
        --out-dir ${OUTPUT}/${L} \
        --verbose
done

INPUT=work/label
OUTPUT=work/label_threshold

for L in ${LABELS[@]}; do
    echo ${L}
    isu analyze threshold \
        --in-dir ${INPUT}/${L} \
        --out-dir ${OUTPUT}/${L} \
        --verbose 
done