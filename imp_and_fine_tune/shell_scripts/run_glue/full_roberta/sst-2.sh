export root_dir=${root_directory}
export task=SST-2
export model_type=roberta

for seed in 1 2 3
do
{
	python $root_dir/imp_and_fine_tune/glue_trans.py \
	       --dir pre \
	       --output_dir $root_dir/imp_and_fine_tune/log/glue/full_$model_type/$task/$seed \
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
	       --seed $seed \
               --cache_dir $root_dir/cache_dir
}
done
