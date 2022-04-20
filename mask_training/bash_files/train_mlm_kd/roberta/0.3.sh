export ROOT_DIR=${root_directory}/mask_training
export mask_dataset=wikitext-103
export EVAL_FILE=$ROOT_DIR/data/$mask_dataset/wiki.valid.raw
export TRAIN_FILE=$ROOT_DIR/data/$mask_dataset/wiki.train.raw
export ZERO_RATE=0.3
export max_seq_len=512
export model_type=roberta
export output_dir=$ROOT_DIR/models/prun_bert/unstructured/train_mlm_kd/$mask_dataset/length$max_seq_len/$model_type/$ZERO_RATE

export seed=1
CUDA_VISIBLE_DEVICES=$(($seed-1)) python $ROOT_DIR/train_mlm.py \
    --model_type $model_type \
    --model_name_or_path roberta-base \
    --teacher_model roberta-base \
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
    --max_steps 6000 \
    --controlled_init magnitude \
    --zero_rate $ZERO_RATE \
    --block_size $max_seq_len \
    --structured false \
    --use_kd true \
    --seed $seed \
    --mlm \
&

export seed=2
CUDA_VISIBLE_DEVICES=$(($seed-1)) python $ROOT_DIR/train_mlm.py \
    --model_type $model_type \
    --model_name_or_path roberta-base \
    --teacher_model roberta-base \
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
    --max_steps 6000 \
    --controlled_init magnitude \
    --zero_rate $ZERO_RATE \
    --block_size $max_seq_len \
    --structured false \
    --use_kd true \
    --seed $seed \
    --mlm \
&

export seed=3
CUDA_VISIBLE_DEVICES=$(($seed-1)) python $ROOT_DIR/train_mlm.py \
    --model_type $model_type \
    --model_name_or_path roberta-base \
    --teacher_model roberta-base \
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
    --max_steps 6000 \
    --controlled_init magnitude \
    --zero_rate $ZERO_RATE \
    --block_size $max_seq_len \
    --structured false \
    --use_kd true \
    --seed $seed \
    --mlm
