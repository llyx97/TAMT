export root_dir=${root_directory}/mask_training
export EVAL_FILE=$root_dir/data/wikitext-103/wiki.valid.raw
export prun_type=train_mlm
export block_size=512

for zero in 0.5 0.6 0.7 0.8
do
	for seed in 1 2 3
	do
		for step in 1000 2000 3000 5000 10000 20000 25000
		do
			python $root_dir/eval_mlm.py \
			    --model_type=bert \
			    --model_name_or_path $root_dir/models/bert_pt \
			    --do_eval \
			    --eval_data_file=$EVAL_FILE \
			    --output_dir $root_dir/log/eval_mlm/unstructured/$prun_type/length$block_size/$zero/seed$seed/step$step \
			    --mask_dir $root_dir/models/prun_bert/unstructured/$prun_type/wikitext-103/length$block_size/$zero/seed$seed/checkpoint-$step/mask.pt  \
			    --load_mlm_head false \
			    --per_gpu_eval_batch_size 16 \
			    --zero_rate $zero \
			    --structured false \
			    --mlm
		done
	done
done
