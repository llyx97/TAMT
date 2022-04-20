export ROOT_DIR=${root_directory}
export prun_type=random
export model_type=roberta

for zero in 0.1 0.2 0.3 0.4
do
	for seed in 1 2 3
	do
		python $ROOT_DIR/imp_and_fine_tune/oneshot.py \
                        --model_type $model_type \
			--model_name_or_path roberta-base \
			--root_dir $ROOT_DIR/imp_and_fine_tune \
			--rate $zero \
			--prun_type $prun_type \
			--seed $seed
	done
done
