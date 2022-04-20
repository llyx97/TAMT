export root_dir=${root_directory}
export task=MNLI

for seed in 1 2 3
do
{
	python $root_dir/imp_and_fine_tune/glue_trans.py \
	       --dir pre \
	       --output_dir $root_dir/imp_and_fine_tune/log/glue/full_bert/$task/$seed \
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
