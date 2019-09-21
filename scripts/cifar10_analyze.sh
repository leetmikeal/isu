set -eu

isu analyze last-precision --result-dir work/output_cifar10_01 --out-path work/output_cifar10_01.csv --sample-init 1000 --sample-step 1000 --verbose
isu analyze last-precision --result-dir work/output_cifar10_02 --out-path work/output_cifar10_02.csv --sample-init 1000 --sample-step 1000 --verbose
isu analyze last-precision --result-dir work/output_cifar10_03 --out-path work/output_cifar10_03.csv --sample-init 1000 --sample-step 1000 --verbose
isu analyze last-precision --result-dir work/output_cifar10_04 --out-path work/output_cifar10_04.csv --sample-init 1000 --sample-step 1000 --verbose
isu analyze last-precision --result-dir work/output_cifar10_05 --out-path work/output_cifar10_05.csv --sample-init 1000 --sample-step 1000 --verbose
