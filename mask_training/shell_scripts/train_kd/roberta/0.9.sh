export ROOT_DIR=${root_directory}/mask_training
export ZERO_RATE=0.9
export rep_loss_type=full_cosine
export max_seq_len=512
export model_type=roberta
export output_dir=$ROOT_DIR/models/prun_bert/unstructured/train_kd/wikitext-103/$model_type/length$max_seq_len/$rep_loss_type/mag_init/$ZERO_RATE

for seed in 1 2 3
do
	python $ROOT_DIR/train_kd.py \
            --model_type $model_type \
	    --teacher_model ${RoBERTa_BASE_DIR}$ \
	    --student_model ${RoBERTa_BASE_DIR}$ \
	    --pregenerated_data $ROOT_DIR/data/wikitext-103-kd/$model_type \
	    --pregenerated_eval_data $ROOT_DIR/data/wikitext-103-kd/$model_type/eval_data \
	    --controlled_init magnitude \
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
done
