export ROOT_DIR=${root_directory}/mask_training
export mask_dataset=wikitext-103
export EVAL_FILE=$ROOT_DIR/data/$mask_dataset/wiki.valid.raw
export TRAIN_FILE=$ROOT_DIR/data/$mask_dataset/wiki.train.raw
export ZERO_RATE=0.7
export max_seq_len=512
export model_type=roberta
export output_dir=$ROOT_DIR/models/prun_bert/unstructured/train_mlm/$mask_dataset/length$max_seq_len/$model_type/training_time2

for seed in 1
do
	python $ROOT_DIR/train_mlm.py \
	    --model_type $model_type \
	    --model_name_or_path roberta-base \
	    --do_train \
	    --do_eval \
	    --eval_data_file $EVAL_FILE \
	    --train_data_file $TRAIN_FILE \
	    --output_dir $output_dir/seed$seed \
	    --output_mask_dir $output_dir/seed$seed \
	    --logging_dir $output_dir/seed$seed/logging \
	    --per_gpu_train_batch_size 16 \
	    --per_gpu_eval_batch_size 16 \
            --num_train_epochs 2 \
            --logging_steps 1000 \
            --save_steps 0 \
	    --controlled_init magnitude \
	    --zero_rate $ZERO_RATE \
            --block_size $max_seq_len \
	    --structured false \
            --seed $seed \
	    --mlm
done
