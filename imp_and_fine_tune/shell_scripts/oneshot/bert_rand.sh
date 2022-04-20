export ROOT_DIR=${root_directory}
export prun_type=random

for zero in 0.1 0.2 0.3 0.4
do
	for seed in 1 2 3
	do
		python $ROOT_DIR/imp_and_fine_tune/oneshot.py \
			--model_name_or_path bert-base-uncased \
			--root_dir $ROOT_DIR/imp_and_fine_tune \
			--rate $zero \
			--prun_type $prun_type \
			--seed $seed
	done
done
