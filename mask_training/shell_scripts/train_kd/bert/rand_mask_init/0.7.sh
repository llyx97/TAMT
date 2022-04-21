export ROOT_DIR=${root_directory}/mask_training
export ZERO_RATE=0.7
export rep_loss_type=full_cosine
export max_seq_len=512
export output_dir=$ROOT_DIR/models/prun_bert/unstructured/train_kd/wikitext-103/length$max_seq_len/$rep_loss_type/rand_mask_init/$ZERO_RATE

export seed=1
CUDA_VISIBLE_DEVICES=$(($seed-1)) python $ROOT_DIR/train_kd.py \
    --teacher_model ${BERT_BASE_DIR}$ \
    --student_model ${BERT_BASE_DIR}$ \
    --pregenerated_data $ROOT_DIR/data/wikitext-103-kd/length$max_seq_len \
    --pregenerated_eval_data $ROOT_DIR/data/wikitext-103-kd/length$max_seq_len/eval_data \
    --controlled_init uniform \
    --root_dir $ROOT_DIR \
    --output_dir $output_dir/seed$seed \
    --output_mask_dir $output_dir/seed$seed \
    --train_batch_size 16 \
    --num_train_epochs 2 \
    --eval_step 100 \
    --save_step 1000 \
    --max_seq_length $max_seq_len \
    --zero_rate $ZERO_RATE \
    --attn_distill False \
    --repr_distill True \
    --structured false \
    --rep_loss_type $rep_loss_type \
    --seed $seed \
&

export seed=2
CUDA_VISIBLE_DEVICES=$(($seed-1)) python $ROOT_DIR/train_kd.py \
    --teacher_model ${BERT_BASE_DIR}$ \
    --student_model ${BERT_BASE_DIR}$ \
    --pregenerated_data $ROOT_DIR/data/wikitext-103-kd/length$max_seq_len \
    --pregenerated_eval_data $ROOT_DIR/data/wikitext-103-kd/length$max_seq_len/eval_data \
    --controlled_init uniform \
    --root_dir $ROOT_DIR \
    --output_dir $output_dir/seed$seed \
    --output_mask_dir $output_dir/seed$seed \
    --train_batch_size 16 \
    --num_train_epochs 2 \
    --eval_step 100 \
    --save_step 1000 \
    --max_seq_length $max_seq_len \
    --zero_rate $ZERO_RATE \
    --attn_distill False \
    --repr_distill True \
    --structured false \
    --rep_loss_type $rep_loss_type \
    --seed $seed \
&

export seed=3
CUDA_VISIBLE_DEVICES=$(($seed-1)) python $ROOT_DIR/train_kd.py \
    --teacher_model ${BERT_BASE_DIR}$ \
    --student_model ${BERT_BASE_DIR}$ \
    --pregenerated_data $ROOT_DIR/data/wikitext-103-kd/length$max_seq_len \
    --pregenerated_eval_data $ROOT_DIR/data/wikitext-103-kd/length$max_seq_len/eval_data \
    --controlled_init uniform \
    --root_dir $ROOT_DIR \
    --output_dir $output_dir/seed$seed \
    --output_mask_dir $output_dir/seed$seed \
    --train_batch_size 16 \
    --num_train_epochs 2 \
    --eval_step 100 \
    --save_step 1000 \
    --max_seq_length $max_seq_len \
    --zero_rate $ZERO_RATE \
    --attn_distill False \
    --repr_distill True \
    --structured false \
    --rep_loss_type $rep_loss_type \
    --seed $seed 
