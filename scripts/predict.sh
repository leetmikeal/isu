set -eu

isu predict --in-dir work/input/ --in-model work/output/trained.h5 --out-dir work/output_predict --batch-size 8 --verbose | tee work/output/predict.log
