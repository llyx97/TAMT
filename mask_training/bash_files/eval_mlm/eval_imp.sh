export root_dir=${root_directory}
export EVAL_FILE=$root_dir/mask_training/data/wikitext-103/wiki.valid.raw
export prun_type=imp_pretrain

for zero in 0.5 0.6 0.7 0.8
do
	for seed in 1 2 3
	do
		for step in 2792
		do
			python $root_dir/mask_training/eval_mlm.py \
			    --model_type=bert \
			    --model_name_or_path bert-base-uncased \
			    --do_eval \
			    --eval_data_file=$EVAL_FILE \
			    --output_dir $root_dir/mask_training/log/eval_mlm/unstructured/$prun_type/$zero/seed$seed/step$step \
			    --mask_dir $root_dir/imp_and_fine_tune/pretrain_prun/imp_pretrain/wikitext-103/prun_step$step/seed$seed/$zero/mask.pt \
			    --load_mlm_head false \
			    --per_gpu_eval_batch_size 16 \
			    --zero_rate $zero \
			    --structured false \
			    --mlm
		done
	done
done
