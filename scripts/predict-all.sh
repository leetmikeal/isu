set -eu

LABELS=(
alcon01
alcon06
alcon07
alcon08
alcon12
)

SCRIPT_DIR=$(cd $(dirname $0); pwd)

for L in ${LABELS[@]}; do
    echo ${L}
    { time sh ${SCRIPT_DIR}/predict.sh ${L} ; } 2>&1 | tee work/predict-${L}.log
done