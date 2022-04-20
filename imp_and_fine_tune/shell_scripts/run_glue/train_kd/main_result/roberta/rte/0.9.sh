export root_dir=${root_directory}
export task=RTE
export zero_rate=0.9
export prun_type=train_kd
export rep_loss_type=full_cosine
export block_size=512
export pretrain_step=21999
export model_type=roberta

for mask_seed in 1 2 3
do
	for seed in 1 2 3
	do
	{
		python $root_dir/imp_and_fine_tune/glue_trans.py \
		       --dir pre \
		       --mask_dir $root_dir/mask_training/models/prun_bert/unstructured/$prun_type/wikitext-103/$model_type/length$block_size/$rep_loss_type/mag_init/$zero_rate/seed$mask_seed/step_$pretrain_step/mask.pt\
		       --output_dir $root_dir/imp_and_fine_tune/log/glue/$prun_type/wikitext-103/$model_type/$rep_loss_type/main_result/length$block_size/$task/$zero_rate/seed$mask_seed/$seed \
		       --logging_steps 50 \
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
