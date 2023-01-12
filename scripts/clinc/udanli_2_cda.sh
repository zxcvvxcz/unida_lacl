

export CUDA_VISIBLE_DEVICES=0


lrs='5e-5 1e-5 5e-6'

seeds='1234 2134 3412 4132'


for seed in  $seeds; do
    for lr in $lrs; do
        python nlp/udanli_v9_cda.py --config configs/nlp/udanli-clinc-cda_0.7.yaml --lr $lr --seed $seed --num_nli_sample 2
    done
done


