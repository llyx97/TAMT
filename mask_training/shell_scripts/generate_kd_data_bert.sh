export root_dir=${root_directory}/mask_training
export length=512

python $root_dir/pregenerate_training_data.py \
	--train_corpus $root_dir/data/wikitext-103/wiki.train.raw \
	--bert_model ${BERT_BASE_DIR}$ \
	--output_dir $root_dir/data/wikitext-103-kd/length$length \
	--max_seq_len $length \
	--do_lower_case \
	--epochs_to_generate 1 \
	--one_seq
	
python $root_dir/pregenerate_training_data.py \
	--train_corpus $root_dir/data/wikitext-103/wiki.valid.raw \
	--bert_model ${BERT_BASE_DIR}$ \
	--output_dir $root_dir/data/wikitext-103-kd/length$length/eval_data \
	--max_seq_len $length \
	--do_lower_case \
	--epochs_to_generate 1 \
	--one_seq
