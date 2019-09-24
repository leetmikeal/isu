set -eu

#isu analyze connection --in-dir work/output_predict/result_images/ --out-dir work/output_predict/analyze_images/ --verbose
#isu analyze pxcm --in-dir1 work/label/alcon01/ --in-dir2 work/label/alcon02/ --out-dir work/output_predict/analyze_images/ --verbose
isu analyze pxcm --in-dir1 work/label/alcon01/ --in-dir2 work/output_predict_190923_edge/alcon01/result_images --out-csv work/output_predict_190923_edge/pxcm_alcon01.csv --verbose
isu analyze pxcm --in-dir1 work/label/alcon06/ --in-dir2 work/output_predict_190923_edge/alcon06/result_images --out-csv work/output_predict_190923_edge/pxcm_alcon06.csv --verbose
isu analyze pxcm --in-dir1 work/label/alcon07/ --in-dir2 work/output_predict_190923_edge/alcon07/result_images --out-csv work/output_predict_190923_edge/pxcm_alcon07.csv --verbose
isu analyze pxcm --in-dir1 work/label/alcon08/ --in-dir2 work/output_predict_190923_edge/alcon08/result_images --out-csv work/output_predict_190923_edge/pxcm_alcon08.csv --verbose
isu analyze pxcm --in-dir1 work/label/alcon12/ --in-dir2 work/output_predict_190923_edge/alcon12/result_images --out-csv work/output_predict_190923_edge/pxcm_alcon12.csv --verbose