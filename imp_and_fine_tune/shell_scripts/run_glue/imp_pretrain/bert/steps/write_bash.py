import os
tasks = 'cola sst-2  sts-b'.split()
steps = [200, 300, 500, 900, 2792]
zeros = '0.6  0.7  0.8'.split()

for task in tasks:
    for zero in zeros:
        for step in steps:
            file = open(os.path.join(task, zero, 'step%s.sh'%step), 'r')
            lines = file.readlines()
            file.close()

            lines[2] = 'export export zero_rate=%s\n'%zero
            file = open(os.path.join(task, zero, 'step%s.sh'%step), 'w')
            for line in lines:
                    file.write(line)
            file.close()
