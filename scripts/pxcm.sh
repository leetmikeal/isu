set -eu

#isu analyze connection --in-dir work/output_predict/result_images/ --out-dir work/output_predict/analyze_images/ --verbose
#isu analyze pxcm --in-dir1 work/label/alcon01/ --in-dir2 work/label/alcon02/ --out-dir work/output_predict/analyze_images/ --verbose

# isu analyze pxcm --in-dir1 work/label/alcon01/ --in-dir2 work/output_predict_190924/alcon01/result_images --out-csv work/output_predict_190924/pxcm_alcon01.csv --verbose
# isu analyze pxcm --in-dir1 work/label/alcon06/ --in-dir2 work/output_predict_190924/alcon06/result_images --out-csv work/output_predict_190924/pxcm_alcon06.csv --verbose
# isu analyze pxcm --in-dir1 work/label/alcon07/ --in-dir2 work/output_predict_190924/alcon07/result_images --out-csv work/output_predict_190924/pxcm_alcon07.csv --verbose
# isu analyze pxcm --in-dir1 work/label/alcon08/ --in-dir2 work/output_predict_190924/alcon08/result_images --out-csv work/output_predict_190924/pxcm_alcon08.csv --verbose
# isu analyze pxcm --in-dir1 work/label/alcon12/ --in-dir2 work/output_predict_190924/alcon12/result_images --out-csv work/output_predict_190924/pxcm_alcon12.csv --verbose

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
TARGET=work/output_ensemble

for L in ${LABELS[@]}; do
    echo ${L}
    isu analyze pxcm \
        --in-dir1 ${MASTER}/${L} \
        --in-dir2 ${TARGET}/${L} \
        --out-csv ${TARGET}/pxcm_${L}.csv \
        --verbose
done
exit