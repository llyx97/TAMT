export root_dir=${root_directory}
export task=MNLI
export zero_rate=0.8
export prun_type=train_kd
export rep_loss_type=full_cosine
export block_size=512
export step=9999

for mask_seed in 1 2 3
do
{
        for seed in 1 2 3
	do
	{
		python $root_dir/imp_and_fine_tune/glue_trans.py \
		       --dir pre \
		       --mask_dir $root_dir/mask_training/models/prun_bert/unstructured/$prun_type/wikitext-103/length$block_size/$rep_loss_type/mag_init/$zero_rate/seed$mask_seed/step_$step/mask.pt\
		       --output_dir $root_dir/imp_and_fine_tune/log/glue/$prun_type/wikitext-103/$rep_loss_type/steps/length$block_size/step$step/$task/$zero_rate/seed$mask_seed/$seed \
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
}
done
