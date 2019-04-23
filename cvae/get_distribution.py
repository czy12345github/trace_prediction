from collections import defaultdict

f = open('test_epoch40', 'r')
result = open('distribution', 'w')
cond = 4
num_of_samples = 200

def prediction(x):
    stats = defaultdict(lambda:0)
    for i in range(num_of_samples):
        line = x.readline().split(' ')
        line[-1] = line[-1][:-1]
        if line==['']:
            print('EOF')
            return False
        stats[line[cond]] += 1
    dis = []
    for x in stats.keys():
        dis.append((x,float(stats[x])/num_of_samples))
    dis = sorted(dis,key=lambda x:x[1],reverse=True)
    newline = []
    for x in dis:
        newline.append(str(x[0]))
        newline.append(str(x[1]))
    result.write(' '.join(newline)+'\n')
    return True

while prediction(f):
    pass

f.close()
result.close()
