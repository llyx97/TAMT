export ROOT_DIR=${root_directory}/mask_training
export EVAL_FILE=$ROOT_DIR/data/wikitext-103/wiki.valid.raw

python $ROOT_DIR/eval_mlm.py \
    --model_type=bert \
    --model_name_or_path bert-base-uncased \
    --do_eval \
    --eval_data_file=$EVAL_FILE \
    --logging_steps=100 \
    --output_dir $ROOT_DIR/log/eval_mlm/full_bert\
    --per_gpu_eval_batch_size 16 \
    --mlm 
