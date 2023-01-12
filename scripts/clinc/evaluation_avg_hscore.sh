

export CUDA_VISIBLE_DEVICES=2


# ####################################################################################################
# method='fine_tuning'

# seed='1234'
# lr='5e-5'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/fine_tuning-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='2134'
# lr='1e-6'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/fine_tuning-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='3412'
# lr='1e-6'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/fine_tuning-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='4132'
# lr='1e-4'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/fine_tuning-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 


# ####################################################################################################
# method='dann'

# seed='1234'
# lr='1e-4'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/dann-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='2134'
# lr='5e-6'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/dann-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='3412'
# lr='5e-6'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/dann-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='4132'
# lr='5e-5'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/dann-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# ####################################################################################################
# method='udalm'

# seed='1234'
# lr='5e-5'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/udalm-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='2134'
# lr='5e-5'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/udalm-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='3412'
# lr='5e-5'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/udalm-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='4132'
# lr='5e-5'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/udalm-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# ####################################################################################################
# method='uan'

# seed='1234'
# lr='1e-4'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/uan-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='2134'
# lr='1e-4'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/uan-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='3412'
# lr='1e-4'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/uan-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='4132'
# lr='1e-5'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/uan-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 


# ####################################################################################################
# method='cmu'

# seed='1234'
# lr='1e-5'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/cmu-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='2134'
# lr='1e-5'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/cmu-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='3412'
# lr='1e-5'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/cmu-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

# seed='4132'
# lr='1e-5'
# python nlp/evaluation_avg_hscore.py --config configs/nlp/cmu-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 


####################################################################################################
method='ovanet'

seed='1234'
lr='5e-5'
python nlp/evaluation_avg_hscore.py --config configs/nlp/ovanet-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

seed='2134'
lr='1e-6'
python nlp/evaluation_avg_hscore.py --config configs/nlp/ovanet-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

seed='3412'
lr='1e-6'
python nlp/evaluation_avg_hscore.py --config configs/nlp/ovanet-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 

seed='4132'
lr='1e-5'
python nlp/evaluation_avg_hscore.py --config configs/nlp/ovanet-clinc-opda.yaml --method_name $method --lr $lr --seed $seed 


# sh scripts/massive/evaluation_avg_hscore.sh