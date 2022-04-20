export root_dir=${root_directory}
export task=SST-2
export prun_type=full_bert
export num_data=30000

for seed in 1 2 3 4 5 
do
{
	python $root_dir/imp_and_fine_tune/glue_trans_few.py \
	       --dir pre \
	       --output_dir $root_dir/imp_and_fine_tune/log/glue/few_shot/$prun_type/$task/num_data$num_data/$seed \
	       --logging_steps 50 \
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
	       --subset_size $num_data \
	       --evaluate_during_training \
	       --save_steps 0 \
	       --seed $seed
}
done
