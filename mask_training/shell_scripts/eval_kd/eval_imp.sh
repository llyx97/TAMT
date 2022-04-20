export root_dir=${root_directory}
export prun_type=imp_pretrain

for ZERO_RATE in 0.5 0.6 0.7 0.8
do
	for seed in 1 2 3
	do
		for step in 2792
		do
			python $root_dir/mask_training/eval_kd.py \
				--do_lower_case \
				--teacher_model bert-base-uncased \
				--student_model bert-base-uncased \
				--pregenerated_eval_data $root_dir/mask_training/data/wikitext-103-kd/eval_data \
				--output_dir $root_dir/mask_training/log/eval_kd/unstructured/$prun_type/$ZERO_RATE/seed$seed/step$step \
				--load_mask_dir $root_dir/imp_and_fine_tune/pretrain_prun/$prun_type/wikitext-103/prun_step$step/seed$seed/$ZERO_RATE/mask.pt \
				--eval_batch_size 16 \
				--repr_distill true \
				--structured false
		done
	done
done
