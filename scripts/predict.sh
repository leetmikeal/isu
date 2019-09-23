set -eu

isu predict --in-dir work/input/alcon01 --in-model work/output/trained.h5 --out-dir work/output_predict/alcon01 --verbose | tee work/output/predict_alcon01.log
isu predict --in-dir work/input/alcon06 --in-model work/output/trained.h5 --out-dir work/output_predict/alcon06 --verbose | tee work/output/predict_alcon06.log
isu predict --in-dir work/input/alcon07 --in-model work/output/trained.h5 --out-dir work/output_predict/alcon07 --verbose | tee work/output/predict_alcon07.log
isu predict --in-dir work/input/alcon08 --in-model work/output/trained.h5 --out-dir work/output_predict/alcon08 --verbose | tee work/output/predict_alcon08.log
isu predict --in-dir work/input/alcon12 --in-model work/output/trained.h5 --out-dir work/output_predict/alcon12 --verbose | tee work/output/predict_alcon12.log
