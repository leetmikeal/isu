set -eu

isu training --in-dir work/ --out-dir work/output --in-weight work/backup_190923-2339/output/trained.h5 --cache-image work/cache/ --epochs 150 --sample-init 300 --sample-val 10 --sample-crop 128 --batch-size 2 --lr-init 0.1 --lr-step 0.1 --lr-epochs 50 --verbose | tee work/output/train.log
