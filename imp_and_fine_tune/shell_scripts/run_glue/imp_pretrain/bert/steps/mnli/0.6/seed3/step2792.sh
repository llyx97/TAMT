export root_dir=${root_directory}
export task=MNLI
export zero_rate=0.6
export prun_type=imp_pretrain
export step=2792
export mask_seed=3

for seed in 1 2 3
do
{
	python $root_dir/imp_and_fine_tune/glue_trans.py \
	       --dir pre \
	       --mask_dir $root_dir/imp_and_fine_tune/pretrain_prun/$prun_type/wikitext-103/prun_step$step/seed$mask_seed/$zero_rate/mask.pt \
	       --output_dir $root_dir/imp_and_fine_tune/log/glue/$prun_type/local/wikitext-103/prun_step$step/$task/$zero_rate/seed$mask_seed/$seed \
               --logging_steps 500 \
	       --task_name $task \
	       --data_dir $root_dir/imp_and_fine_tune/glue/$task \
	       --model_type bert \
	       --model_name_or_path bert-base-uncased \
	       --do_train \
	       --do_eval \
	       --do_lower_case \
	       --max_seq_length 128 \
	       --per_gpu_train_batch_size 32 \
	       --learning_rate 2e-5 \
	       --num_train_epochs 3 \
	       --evaluate_during_training \
	       --save_steps 0 \
	       --seed $seed
}
done
