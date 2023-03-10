

export CUDA_VISIBLE_DEVICES=2



lrs='4e-4'

# seeds='2134'
seeds='1234 2134 3412 4132'

# # B -> D, E, K

# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#     python3 nlp/adspt_no_verbalizer.py --config configs/nlp/adspt-amazon-books-dvd-elara-pch.yaml --seed $seed --lr $lr
#     done
# done
# CDA
for seed in  $seeds; do
    for lr in $lrs; do
    python3 nlp/adspt_no_verbalizer.py --config configs/nlp/adspt-amazon-books-electronics-elara-pch.yaml --seed $seed --lr $lr
    done
done

# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#     python3 nlp/adspt_no_verbalizer.py --config configs/nlp/adspt-amazon-books-kitchen-elara-pch.yaml --seed $seed --lr $lr
#     done
# done


# # D -> B, E, K

# CDA
for seed in  $seeds; do
    for lr in $lrs; do
    python3 nlp/adspt_mlm_only.py --config configs/nlp/adspt-amazon-dvd-books-elara-pch.yaml --seed $seed 
    done
done

# CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python3 nlp/adspt.py --config configs/nlp/adspt-amazon-dvd-electronics-elara-pch.yaml --lr $lr --seed $seed
#     done
# done


# CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/adspt.py --config configs/nlp/adspt-amazon-dvd-kitchen-elara-pch.yaml --lr $lr --seed $seed
#     done
# done




# # E -> B, D, K

# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/cmu.py --config configs/nlp/cmu-amazon-electronics-books.yaml --lr $lr --seed $seed
#     done
# done

# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/cmu.py --config configs/nlp/cmu-amazon-electronics-dvd.yaml --lr $lr --seed $seed
#     done
# done


# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/cmu.py --config configs/nlp/cmu-amazon-electronics-kitchen.yaml --lr $lr --seed $seed
#     done
# done


# # K -> B, D, E

# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/cmu.py --config configs/nlp/cmu-amazon-kitchen-books.yaml --lr $lr --seed $seed
#     done
# done

# CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/cmu.py --config configs/nlp/cmu-amazon-kitchen-dvd.yaml --lr $lr --seed $seed
#     done
# done


# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/cmu.py --config configs/nlp/cmu-amazon-kitchen-electronics.yaml --lr $lr --seed $seed
#     done
# done
