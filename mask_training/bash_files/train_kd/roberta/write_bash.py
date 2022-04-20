import os
zero_rates = ['0.5', '0.6', '0.7', '0.8', '0.9']

for zero in zero_rates:
	file = open('%s.sh'%zero, 'r')
	lines = file.readlines()
	file.close()
	
	lines[1] = 'export ZERO_RATE=%s\n'%zero
	#lines[17] = '            --num_train_epochs 1 \\\n'
	file = open('%s.sh'%zero, 'w')
	for line in lines:
		file.write(line)
	file.close()
