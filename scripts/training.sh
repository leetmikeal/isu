set -eu

isu training --in-dir work/ --out-dir work/output --cache-image work/cache/ --epochs 90 --sample-init 400 --sample-val 100 --batch-size 8 --lr-init 0.1 --lr-step 0.1 --lr-epochs 30 --verbose
