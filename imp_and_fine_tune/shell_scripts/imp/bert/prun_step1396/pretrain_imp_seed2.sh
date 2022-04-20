export ROOT_DIR=${root_directory}
export seed=2

python $ROOT_DIR/imp_and_fine_tune/LT_pretrain.py \
           --output_dir $ROOT_DIR/imp_and_fine_tune/pretrain_prun/imp_pretrain/wikitext-103/prun_step1396/seed$seed \
           --model_type bert \
           --model_name_or_path bert-base-uncased \
           --train_data_file $ROOT_DIR/mask_training/data/wikitext-103/wiki.train.raw \
           --do_train \
           --eval_data_file $ROOT_DIR/mask_training/data/wikitext-103/wiki.valid.raw \
           --do_eval \
           --per_gpu_train_batch_size 16 \
           --per_gpu_eval_batch_size 16 \
           --evaluate_during_training \
           --num_train_epochs 1 \
           --logging_steps 1396 \
           --save_steps 1396 \
           --mlm \
           --seed $seed
