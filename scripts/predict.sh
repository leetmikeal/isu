set -eu

isu predict --in-dir work/input/alcon01 --in-model work/output/trained.h5 --out-dir work/output_predict --verbose | tee work/output/predict.log
