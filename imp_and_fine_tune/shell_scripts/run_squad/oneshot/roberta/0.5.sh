export root_dir=${root_directory}
export task=squad
export zero_rate=0.5
export prun_type=oneshot
export data_dir=$root_dir/mask_training/data/squad
export model_type=roberta

for seed in 1 2 3
do
{
python $root_dir/imp_and_fine_tune/squad_trans.py \
           --dir pre \
           --mask_dir $root_dir/imp_and_fine_tune/pretrain_prun/$prun_type/magnitude/$model_type/$zero_rate/mask.pt \
           --output_dir $root_dir/imp_and_fine_tune/log/glue/$prun_type/$model_type/$task/$zero_rate/$seed \
           --model_type $model_type \
           --model_name_or_path roberta-base \
           --do_train \
           --do_eval \
           --data_dir $data_dir \
           --train_file $data_dir/train-v1.1.json \
           --predict_file $data_dir/dev-v1.1.json \
           --per_gpu_train_batch_size 16 \
           --learning_rate 3e-5 \
           --num_train_epochs 2 \
           --max_seq_length 384 \
           --doc_stride 128 \
           --evaluate_during_training \
           --eval_all_checkpoints \
           --logging_steps 1000 \
           --save_steps 0 \
           --seed $seed
}
done
