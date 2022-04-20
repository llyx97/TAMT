export root_dir=${root_directory}
export prun_type=random

for ZERO_RATE in 0.5 0.6 0.7 0.8 0.9
do
	for seed in 1 2 3
	do
		python $root_dir/mask_training/eval_kd.py \
			--do_lower_case \
			--teacher_model bert-base-uncased \
			--student_model bert-base-uncased \
			--pregenerated_eval_data $root_dir/mask_training/data/wikitext-103-kd/eval_data \
			--output_dir $root_dir/mask_training/log/eval_kd/unstructured/$prun_type/$ZERO_RATE/seed$seed \
                        --load_mask_dir $root_dir/imp_and_fine_tune/pretrain_prun/oneshot/$prun_type/bert/$ZERO_RATE/$seed/mask.pt \
			--eval_batch_size 16 \
			--repr_distill true \
			--structured false
	done
done
