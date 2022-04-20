import os
zero_rates = ['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']

for zero in zero_rates:
	file = open('%s.sh'%zero, 'r')
	lines = file.readlines()
	file.close()
	
	lines[4] = 'export ZERO_RATE=%s\n'%zero
	lines[5] = 'export max_seq_len=512\n'
	lines[23] = '            --num_train_epochs 2 \\\n'
	lines[24] = '            --logging_steps 1000 \\\n'
	lines[25] = '            --save_steps 1000 \\\n'
	file = open('%s.sh'%zero, 'w')
	for line in lines:
		file.write(line)
	file.close()
