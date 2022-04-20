export root_dir=${root_directory}
export EVAL_FILE=$root_dir/mask_training/data/wikitext-103/wiki.valid.raw
export ZERO_RATE=0.9
export prun_type=random

for seed in 1 2 3
do
		python $root_dir/mask_training/eval_mlm.py \
		    --model_type=bert \
		    --model_name_or_path bert-base-uncased \
		    --do_eval \
		    --eval_data_file=$EVAL_FILE \
		    --output_dir $root_dir/mask_training/log/eval_mlm/unstructured/$prun_type/$ZERO_RATE/seed$seed \
                    --mask_dir $root_dir/imp_and_fine_tune/pretrain_prun/oneshot/$prun_type/bert/$ZERO_RATE/$seed/mask.pt \
		    --load_mlm_head false \
		    --per_gpu_eval_batch_size 16 \
		    --zero_rate $ZERO_RATE \
		    --structured false \
		    --mlm
done
