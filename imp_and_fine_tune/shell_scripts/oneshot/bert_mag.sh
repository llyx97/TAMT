export ROOT_DIR=${root_directory}
export model_type=bert

for zero in 0.1
do
	python $ROOT_DIR/imp_and_fine_tune/oneshot.py \
		--model_type $model_type \
		--model_name_or_path bert-base-uncased \
		--root_dir $ROOT_DIR/imp_and_fine_tune \
		--rate $zero 
done
