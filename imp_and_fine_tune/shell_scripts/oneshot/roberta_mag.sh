export ROOT_DIR=${root_directory}
export model_type=roberta

for zero in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
	python $ROOT_DIR/imp_and_fine_tune/oneshot.py \
		--model_type $model_type \
		--model_name_or_path roberta-base \
		--root_dir $ROOT_DIR/imp_and_fine_tune \
		--rate $zero 
done
