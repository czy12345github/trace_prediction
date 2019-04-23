import pickle
import random

testtrace_min_length = 9

def random_select(total, num):
    selected = []
    while len(selected) < num:
        temp = random.randint(0, total-1)
        if temp not in selected:
            selected.append(temp)
    return selected

def get_trainset_testset(traces):
    trainset = []
    testset = []
    candidates = []

    for trace in traces:
        if len(trace) < testtrace_min_length:
            trainset.append(trace)
        else:
            candidates.append(trace)

    testsize = len(candidates)/3

    selected = random_select(len(candidates), testsize)
    for i in range(len(candidates)):
        if i in selected:
            testset.append(candidates[i])
        else:
            trainset.append(candidates[i])

    return trainset, testset

if __name__ == '__main__':
    f = open('traces3', 'rb')
    traces = pickle.load(f)
    f.close()
    train, test = get_trainset_testset(traces)
    f = open('train_traces', 'wb')
    pickle.dump(train, f)
    f.close()
    f = open('test_traces', 'wb')
    pickle.dump(test, f)
    f.close()
