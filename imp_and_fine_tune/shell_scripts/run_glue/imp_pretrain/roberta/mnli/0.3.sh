export root_dir=${root_directory}
export task=MNLI
export zero_rate=0.3
export prun_type=imp_pretrain
export pretrain_step=2792
export model_type=roberta

for mask_seed in 1 2 3
do
	for seed in 1 2 3
	do
	{
		python $root_dir/imp_and_fine_tune/glue_trans.py \
		       --dir pre \
                       --mask_dir $root_dir/imp_and_fine_tune/pretrain_prun/$prun_type/wikitext-103/$model_type/prun_step$pretrain_step/seed$mask_seed/$zero_rate/mask.pt \
                       --output_dir $root_dir/imp_and_fine_tune/log/glue/$prun_type/local/wikitext-103/$model_type/main_result/$task/$zero_rate/seed$mask_seed/$seed \
                       --logging_steps 500 \
		       --task_name $task \
		       --data_dir $root_dir/imp_and_fine_tune/glue/$task \
		       --model_type $model_type \
		       --model_name_or_path roberta-base \
		       --do_train \
		       --do_eval \
		       --max_seq_length 128 \
		       --per_gpu_train_batch_size 32 \
		       --learning_rate 2e-5 \
		       --num_train_epochs 3 \
		       --evaluate_during_training \
		       --save_steps 0 \
		       --seed $seed
	}
	done
done
