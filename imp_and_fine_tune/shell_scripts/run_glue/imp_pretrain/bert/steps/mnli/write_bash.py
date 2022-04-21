import os
seeds = 'seed1  seed2  seed3'.split()
steps = [200, 300, 500, 900, 2792]
zeros = '0.6  0.7  0.8'.split()

for seed in seeds:
    for zero in zeros:
        for step in steps:
            file = open(os.path.join(zero, seed, 'step%s.sh'%step), 'r')
            lines = file.readlines()
            file.close()

            lines[2] = 'export zero_rate=%s\n'%zero
            file = open(os.path.join(zero, seed, 'step%s.sh'%step), 'w')
            for line in lines:
                    file.write(line)
            file.close()
