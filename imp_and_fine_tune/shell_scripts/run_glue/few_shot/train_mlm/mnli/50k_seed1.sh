export root_dir=${root_directory}
export task=MNLI
export zero_rate=0.7
export prun_type=train_mlm
export num_data=50000

for mask_seed in 1
do
        for seed in 1 2 3 4 5 
	do
	{
		python $root_dir/imp_and_fine_tune/glue_trans_few.py \
		       --dir pre \
                       --mask_dir $root_dir/mask_training/models/prun_bert/unstructured/$prun_type/wikitext-103/length512/$zero_rate/seed$mask_seed/checkpoint-17000/mask.pt \
                       --output_dir $root_dir/imp_and_fine_tune/log/glue/few_shot/$prun_type/length512/$task/$zero_rate/num_data$num_data/seed$mask_seed/$seed \
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
                       --subset_size $num_data \
		       --evaluate_during_training \
		       --save_steps 0 \
		       --seed $seed
	}
	done
done
