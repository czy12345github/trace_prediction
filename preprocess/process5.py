import pickle
import numpy as np

input_size = 5

def remove_sojourn(traces):
    new_traces = []
    for trace in traces:
        new_trace = []
        for x in trace:
            new_trace.append(x[0])
        new_traces.append(new_trace)
    return new_traces

def get_onehot_trace(trace, total_scs):
    new_trace = []
    for x in trace:
        temp = np.zeros(total_scs)
        temp[x-1] = 1
        new_trace.extend(list(temp))
    return new_trace

def transfer_traces(traces, total_scs):
    new_traces = []

    for trace in traces:
        length = len(trace)
        num = length - input_size + 1
        for i in range(num):
            new_traces.append(get_onehot_trace(trace[i:i+input_size], total_scs))

    return new_traces

def matlab_transfer_traces(traces):
    new_traces = []

    for trace in traces:
        length = len(trace)
        num = length - input_size + 1
        for i in range(num):
            new_traces.append(trace[i:i+input_size])

    return new_traces

def write_onehot_traces(traces, total_scs, result):
    traces = remove_sojourn(traces)
    traces = transfer_traces(traces, total_scs)

    for trace in traces:
        line = [str(x) for x in trace]
        line = ' '.join(line) + '\n'
        result.write(line)

def write_traces(traces, result):
    traces = remove_sojourn(traces)
    traces = matlab_transfer_traces(traces)

    for trace in traces:
        line = [str(x) for x in trace]
        line = ' '.join(line) + '\n'
        result.write(line)

if __name__ == '__main__':
    f = open('train_traces', 'rb')
    train_traces = pickle.load(f)
    f.close()
    f = open('test_traces', 'rb')
    test_traces = pickle.load(f)
    f.close()
    f = open('index_smallcell', 'rb')
    index_smallcell = pickle.load(f)
    f.close()
    total_scs = len(index_smallcell)
    f = open('trainset', 'w')
    write_onehot_traces(train_traces, total_scs, f)
    f.close()
    f = open('testset', 'w')
    write_onehot_traces(test_traces, total_scs, f)
    f.close()
    f = open('matlab_trainset', 'w')
    write_traces(train_traces, f)
    f.close()
    f = open('matlab_testset', 'w')
    write_traces(test_traces, f)
    f.close()
