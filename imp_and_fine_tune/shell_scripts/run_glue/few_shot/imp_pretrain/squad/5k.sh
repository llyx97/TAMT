export root_dir=${root_directory}
export task=squad
export zero_rate=0.7
export prun_type=imp_pretrain
export data_dir=$root_dir/mask_training/data/squad
export num_data=5000

for mask_seed in 1 2 3
do
	for seed in 1 2 3 4 5
	do
	{
	python $root_dir/imp_and_fine_tune/squad_trans_few.py \
		   --dir pre \
                   --mask_dir $root_dir/imp_and_fine_tune/pretrain_prun/$prun_type/wikitext-103/prun_step2792/seed$mask_seed/$zero_rate/mask.pt \
                   --output_dir $root_dir/imp_and_fine_tune/log/glue/few_shot/$prun_type/prun_step2792/$task/$zero_rate/num_data$num_data/seed$mask_seed/$seed \
		   --model_type bert \
		   --model_name_or_path bert-base-uncased \
		   --do_train \
		   --do_eval \
		   --do_lower_case \
                   --data_dir $data_dir \
		   --train_file $data_dir/train-v1.1.json \
		   --predict_file $data_dir/dev-v1.1.json \
		   --per_gpu_train_batch_size 16 \
		   --learning_rate 3e-5 \
		   --num_train_epochs 2 \
		   --max_seq_length 384 \
		   --doc_stride 128 \
                   --subset_size $num_data \
		   --evaluate_during_training \
		   --eval_all_checkpoints \
                   --logging_steps 500 \
		   --save_steps 0 \
		   --seed $seed
	}
	done
done
