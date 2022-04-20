export ROOT_DIR=${root_directory}/mask_training
export mask_dataset=wikitext-103
export EVAL_FILE=$ROOT_DIR/data/$mask_dataset/wiki.valid.raw
export TRAIN_FILE=$ROOT_DIR/data/$mask_dataset/wiki.train.raw
export ZERO_RATE=0.7
export max_seq_len=512
export output_dir=$ROOT_DIR/models/prun_bert/unstructured/train_mlm/$mask_dataset/length$max_seq_len/rand_mask_init/$ZERO_RATE

export seed=1
CUDA_VISIBLE_DEVICES=$(($seed-1)) python $ROOT_DIR/train_mlm.py \
    --model_type=bert \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --eval_data_file $EVAL_FILE \
    --train_data_file $TRAIN_FILE \
    --output_dir $output_dir/seed$seed \
    --output_mask_dir $output_dir/seed$seed \
    --logging_dir $output_dir/seed$seed/logging \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --num_train_epochs 2 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --controlled_init uniform \
    --zero_rate $ZERO_RATE \
    --block_size $max_seq_len \
    --structured false \
    --seed $seed \
    --mlm \
    --logging_first_step \
&

export seed=2
CUDA_VISIBLE_DEVICES=$(($seed-1)) python $ROOT_DIR/train_mlm.py \
    --model_type=bert \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --eval_data_file $EVAL_FILE \
    --train_data_file $TRAIN_FILE \
    --output_dir $output_dir/seed$seed \
    --output_mask_dir $output_dir/seed$seed \
    --logging_dir $output_dir/seed$seed/logging \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --num_train_epochs 2 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --controlled_init uniform \
    --zero_rate $ZERO_RATE \
    --block_size $max_seq_len \
    --structured false \
    --seed $seed \
    --mlm \
    --logging_first_step \
&

export seed=3
CUDA_VISIBLE_DEVICES=$(($seed-1)) python $ROOT_DIR/train_mlm.py \
    --model_type=bert \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --eval_data_file $EVAL_FILE \
    --train_data_file $TRAIN_FILE \
    --output_dir $output_dir/seed$seed \
    --output_mask_dir $output_dir/seed$seed \
    --logging_dir $output_dir/seed$seed/logging \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 16 \
    --num_train_epochs 2 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --controlled_init uniform \
    --zero_rate $ZERO_RATE \
    --block_size $max_seq_len \
    --structured false \
    --seed $seed \
    --mlm \
    --logging_first_step \
