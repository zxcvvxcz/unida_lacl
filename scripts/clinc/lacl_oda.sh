export CUDA_VISIBLE_DEVICES=2



lrs='1e-5'
# lrs='4e-4'

seeds='4132'
# seeds='1234 2134 3412 4132'

# # B -> D, E, K

# CDA
for seed in  $seeds; do
    for lr in $lrs; do
    python3 nlp/lacl_oda.py --config configs/nlp/lacl-clinc-oda-elara-pch.yaml --seed $seed --lr $lr
    done
done