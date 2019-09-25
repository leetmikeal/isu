set -eu

if [ $# -ne 1 ]; then
    echo "not found argument" 1>&2
    echo "  sh predict.sh [dataset name]" 1>&2
    echo "  ex) sh predict.sh alcon01" 1>&2
    exit 1
fi
L=$1
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
    