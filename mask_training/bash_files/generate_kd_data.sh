export root_dir=${root_directory}/mask_training

for length in 512
do
	python $root_dir/pregenerate_training_data.py \
		--train_corpus $root_dir/data/wikitext-103/wiki.train.raw \
		--bert_model $root_dir/models/bert_pt \
		--output_dir $root_dir/data/wikitext-103-kd/length$length \
		--max_seq_len $length \
		--epochs_to_generate 1 
done
