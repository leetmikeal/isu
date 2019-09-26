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
    isu analyze collision \
        --in-dir1 work/label/${L} \
        --in-dir2 work/output/${L} \
        --out-csv work/output/${L}_collision.csv \
        --verbose
done