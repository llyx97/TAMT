import os

filenames = []
def processDirectory(args, dirname, filenames_):
	for filename in filenames_:
		filenames.append(os.path.join(dirname, filename))
os.path.walk('.', processDirectory, None)

strings_to_replace = {
	'/apdcephfs/share_47076/tmpv_xiuliu': '${root_directory}',
	'$ROOT_DIR/models/bert_pt': 'bert-base-uncased', 
	'$root_dir/LT/models/bert_pt': 'bert-base-uncased',
	'$ROOT_DIR/models/$model_type': 'roberta-base',
	'LT': 'mask_training',
	'bert-lt': 'imp_and_fine_tune',
	'wikitext-2': 'wikitext-103'
}

for filename in filenames:
	if not filename.endswith('.sh'):
		continue
	file = open(filename, 'r')
        lines = file.readlines()
        file.close()

	for i, line in enumerate(lines):
		#if 'export ROOT_DIR=' in line:
		#	lines[i] = 'export ROOT_DIR=${root_directory}\n'
		for string in strings_to_replace:
			if string in line:
				lines[i] = lines[i].replace(string, strings_to_replace[string])

	file = open(filename, 'w')
        for line in lines:
                file.write(line)
        file.close()
