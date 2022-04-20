export ROOT_DIR=${root_directory}
export prun_step=2792
export seed=1
export model_type=roberta

python $ROOT_DIR/imp_and_fine_tune/LT_pretrain.py \
           --output_dir $ROOT_DIR/imp_and_fine_tune/pretrain_prun/imp_pretrain/wikitext-103/$model_type/prun_step$prun_step/training_time \
           --model_type $model_type \
           --model_name_or_path roberta-base \
           --train_data_file $ROOT_DIR/mask_training/data/wikitext-103/wiki.train.raw \
           --do_train \
           --eval_data_file $ROOT_DIR/mask_training/data/wikitext-103/wiki.valid.raw \
           --per_gpu_train_batch_size 16 \
           --per_gpu_eval_batch_size 16 \
           --num_train_epochs 2 \
           --logging_steps $prun_step \
           --save_steps $prun_step \
           --mlm \
           --seed $seed
