

export CUDA_VISIBLE_DEVICES=0



lrs='1e-5'
# lrs='1e-5 5e-6 1e-6'

seeds='1234'
# seeds='1234 2134 3412 4132'
# class_ratios='0.2 0.5 1 2 5'
class_ratios='5'

# # B -> D, E, K

# CDA
for ratio in $class_ratios; do
    for seed in  $seeds; do
        for lr in $lrs; do
        python3 nlp/adspt_ft.py --config configs/nlp/adspt-amazon-dvd-books-elara-pch.yaml --seed $seed
        done
    done
done

# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/cmu.py --config configs/nlp/cmu-amazon-books-electronics.yaml --lr $lr --seed $seed
#     done
# done


# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/cmu.py --config configs/nlp/cmu-amazon-books-kitchen.yaml --lr $lr --seed $seed
#     done
# done


# # D -> B, E, K

# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/cmu.py --config configs/nlp/cmu-amazon-dvd-books.yaml --lr $lr --seed $seed
#     done
# done

# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/cmu.py --config configs/nlp/cmu-amazon-dvd-electronics.yaml --lr $lr --seed $seed
#     done
# done


# # CDA
# for seed in  $seeds; do
#     for lr in $lrs; do
#         python nlp/cmu.py --config configs/nlp/cmu-amazon-dvd-kitchen.yaml --lr $lr --seed $seed
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
