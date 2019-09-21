set -eu

# isu training --in-dir work/cifar10 --cache-image work/cache_cifar10 --out-dir work/output_cifar10_01 --epochs 120 --batch-size 256 --sample-init 1000 --sample-step 1000 --sample-val 10000 --lr-init 0.1 --lr-step 0.1 --lr-epochs 30 --verbose
# isu training --in-dir work/cifar10 --cache-image work/cache_cifar10 --out-dir work/output_cifar10_02 --epochs 120 --batch-size 256 --sample-init 1000 --sample-step 1000 --sample-val 10000 --lr-init 0.1 --lr-step 0.1 --lr-epochs 30 --verbose
# isu training --in-dir work/cifar10 --cache-image work/cache_cifar10 --out-dir work/output_cifar10_03 --epochs 120 --batch-size 256 --sample-init 1000 --sample-step 1000 --sample-val 10000 --lr-init 0.1 --lr-step 0.1 --lr-epochs 30 --verbose
# isu training --in-dir work/cifar10 --cache-image work/cache_cifar10 --out-dir work/output_cifar10_04 --epochs 120 --batch-size 256 --sample-init 1000 --sample-step 1000 --sample-val 10000 --lr-init 0.1 --lr-step 0.1 --lr-epochs 30 --verbose
# isu training --in-dir work/cifar10 --cache-image work/cache_cifar10 --out-dir work/output_cifar10_05 --epochs 120 --batch-size 256 --sample-init 1000 --sample-step 1000 --sample-val 10000 --lr-init 0.1 --lr-step 0.1 --lr-epochs 30 --verbose

# isu training --in-dir work/cifar10 --cache-image work/cache_cifar10 --out-dir work/output_cifar10_confident_01 --epochs 120 --batch-size 256 --sample-add confident --sample-init 1000 --sample-step 1000 --sample-val 10000 --lr-init 0.1 --lr-step 0.1 --lr-epochs 30 --verbose
# isu training --in-dir work/cifar10 --cache-image work/cache_cifar10 --out-dir work/output_cifar10_confident_02 --epochs 120 --batch-size 256 --sample-add confident --sample-init 1000 --sample-step 1000 --sample-val 10000 --lr-init 0.1 --lr-step 0.1 --lr-epochs 30 --verbose
# isu training --in-dir work/cifar10 --cache-image work/cache_cifar10 --out-dir work/output_cifar10_confident_03 --epochs 120 --batch-size 256 --sample-add confident --sample-init 1000 --sample-step 1000 --sample-val 10000 --lr-init 0.1 --lr-step 0.1 --lr-epochs 30 --verbose

isu training --in-dir work/cifar10 --cache-image work/cache_cifar10 --out-dir work/output_cifar10_unconfident_01 --epochs 120 --batch-size 256 --sample-add unconfident --sample-init 1000 --sample-step 1000 --sample-val 10000 --lr-init 0.1 --lr-step 0.1 --lr-epochs 30 --verbose