from collections import defaultdict
import pickle

f = open('distribution', 'r')
t = open('../test_traces', 'rb')
test_traces = pickle.load(t)
t.close()
total = defaultdict(lambda:0)
correct = defaultdict(lambda:0)
cond = 4

predicts = []
targets = []

interval = 0.02

def section(pro):
    for i in range(int(1/interval)):
        if pro > 1-(i+1)*interval and pro <= 1-i*interval:
            return i

def print_result():
    sum_total = 0
    sum_correct = 0
    for x in total.keys():
        sum_total += total[x]
        sum_correct += correct[x]
    for i in range(int(1/interval)):
        if total[i] != 0:
            correct_rate = float(correct[i])/total[i]
            print('%.2f~%.2f,total: %d,correct rate: %.3f' % ((1-(i+1)*interval),(1-i*interval),total[i],correct_rate))
    print('all: %d, overall correct rate: %.3f' % (sum_total, float(sum_correct)/sum_total))

def read_predicts():
    line = f.readline()
    while line:
        temp = line.split(' ')
        temp[-1] = temp[-1][:-1]
        predicts.append((int(temp[0]),float(temp[1])))
        line = f.readline()
    f.close()

def read_targets():
    for trace in test_traces:
        for x in trace[cond:]:
            targets.append(x[0])

def analysis():
    for i in range(len(predicts)):
        sec = section(predicts[i][1])
        total[sec] += 1
        if predicts[i][0] == targets[i]:
            correct[sec] += 1

read_predicts()
read_targets()
analysis()
print_result()
