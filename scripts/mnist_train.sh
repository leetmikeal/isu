set -eu

isu training --in-dir work/mnist --cache-image work/cache_mnist --out-dir work/output_mnist_random_01 --epochs 120 --batch-size 256 --sample-add random --sample-init 1000 --sample-step 1000 --sample-val 10000 --sample-color gray --lr-init 0.1 --lr-step 0.1 --lr-epochs 30 --verbose