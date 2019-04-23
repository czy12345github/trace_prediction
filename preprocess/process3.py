import pickle
from collections import defaultdict

threshold = 10
min_length = 5

def get_sc_occurrence(traces):
    sc_occurrence = defaultdict(lambda: 0)

    for trace in traces:
        for point in trace:
            sc_occurrence[point[0]] += 1
    return sc_occurrence

def cells_info(traces):
    sc_occurrence = get_sc_occurrence(traces)

    below_threshold = 0
    for x in sc_occurrence.keys():
        if sc_occurrence[x] < threshold:
            below_threshold += 1

    print('total small cells: %d' % len(sc_occurrence.keys()))
    print('number of small cells occurred less than %d times: %d' % (threshold, below_threshold))

def find_replacement(trace, i, valid_scs, scs):
    if i == len(trace)-1:
        return None
    for sc in valid_scs:
        if scs.is_neighbor(trace[i-1][0], sc) and scs.is_neighbor(trace[i+1][0], sc) and sc != trace[i-1][0] and sc != trace[i+1][0]:
            return (sc,trace[i][1])
    return None

def _remove_disturb_scs(traces, scs):
    sc_occurrence = get_sc_occurrence(traces)
    valid_scs = []
    invalid_scs = []
    for x in sc_occurrence.keys():
        if sc_occurrence[x] < threshold:
            invalid_scs.append(x)
        else:
            valid_scs.append(x)

    new_traces = []
    for trace in traces:
        pre_index = None
        for i in range(len(trace)):
            if trace[i][0] in valid_scs:
                if pre_index == None:
                    pre_index = i
                    continue
                else:
                    continue
            else:
                if pre_index == None:
                    continue
                else:
                    point  = find_replacement(trace, i, valid_scs, scs)
                    if point == None:
                        if len(trace[pre_index:i]) >= min_length:
                            new_traces.append(trace[pre_index:i])
                        pre_index = None
                    else:
                        trace[i] = point
        if pre_index != None and len(trace[pre_index:]) >= min_length:
            new_traces.append(trace[pre_index:])
    return new_traces

def have_disturb_scs(traces):
    sc_occurrence = get_sc_occurrence(traces)
    for x in sc_occurrence.keys():
        if sc_occurrence[x] < threshold:
            return True
    return False

def remove_disturb_scs(traces, scs):
    new_traces = _remove_disturb_scs(traces, scs)
    #print new_traces
    while have_disturb_scs(new_traces):
        new_traces = _remove_disturb_scs(traces, scs)
    return new_traces

def traces_map(traces):
    smallcell_index = {}

    i = 1
    for x in get_sc_occurrence(traces).keys():
        smallcell_index[x] = i
        i += 1

    new_traces = []
    for trace in traces:
        new_trace = []
        for point in trace:
            new_trace.append((smallcell_index[point[0]],point[1]))
        new_traces.append(new_trace)

    index_smallcell = []
    for i in range(len(smallcell_index.keys())):
        for x in smallcell_index.keys():
            if smallcell_index[x] == i+1:
                index_smallcell.append(x)
                break

    return index_smallcell, new_traces

if __name__ == '__main__':
    f = open('traces2', 'rb')
    traces = pickle.load(f)
    f.close()
    cells_info(traces)
    f = open('scs', 'rb')
    scs = pickle.load(f)
    f.close()

    traces = remove_disturb_scs(traces, scs)
    cells_info(traces)

    index_smallcell, traces = traces_map(traces)
    f = open('traces3', 'wb')
    pickle.dump(traces, f)
    f.close()
    f = open('index_smallcell', 'wb')
    pickle.dump(index_smallcell, f)
    f.close()
