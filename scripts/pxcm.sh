set -eu

#isu analyze connection --in-dir work/output_predict/result_images/ --out-dir work/output_predict/analyze_images/ --verbose
isu analyze pxcm --in-dir1 work/label/alcon01/ --in-dir2 work/label/alcon02/ --out-dir work/output_predict/analyze_images/ --verbose