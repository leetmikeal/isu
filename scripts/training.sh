set -eu

isu training --in-dir work/ --out-dir work/output --cache-image work/cache/ --epochs 90 --sample-init 100 --sample-val 10 --batch-size 2 --lr-init 0.1 --lr-step 0.1 --lr-epochs 30 --verbose | tee work/output/train.log
