import pickle
from collections import defaultdict
import random
import math


min_sojourn = 20
max_sojourn = 200
min_length = 5

sojourn_changes = 0

#sc -> (sojourn mean, sojourn_std)
referrence_mean_std = defaultdict(lambda: (0,0))

def compute_mean_std(data):
    mean = sum(data)*1.0/len(data)

    std = 0.0
    for x in data:
        std += (float(x)-mean)**2
    std = math.sqrt(std/len(data))
    return (mean, std)

def compute_referrence_mean_std(traces):
    global referrence_mean_std

    valid_sojourn = defaultdict(lambda: [])

    for trace in traces:
        for i in range(len(trace)):
            sojourn = trace[i][1]
            if sojourn >= min_sojourn and sojourn <= max_sojourn:
                valid_sojourn[trace[i][0]].append(sojourn)

    for x in valid_sojourn.keys():
        referrence_mean_std[x] = compute_mean_std(valid_sojourn[x])

def traces_info(traces):
    shorts = 0 # number of traces with length less than 5
    longs = 0 # number of traces with length more than 9

    below_min_so = 0 # number of points with sojourn time less than min_sojourn
    above_max_so = 0
    total = 0 # total points

    for trace in traces:
        length = len(trace)
        if length < 5:
            shorts += 1
        elif length >= 9:
            longs += 1
        for x in trace:
            total += 1
            if x[1] < min_sojourn:
                below_min_so += 1
            elif x[1] > max_sojourn:
                above_max_so += 1

    print('number of traces: %d' % len(traces))
    print('number of traces with length less than 5: %d' % shorts)
    print('number of traces with length more than 9: %d' % longs)

    print('total points: %d' % total)
    print('number of points with sojourn time less than %d: %d' % (min_sojourn, below_min_so))
    print('number of points with sojourn time more than %d: %d' % (max_sojourn, above_max_so))

def remove_short_traces(traces):
    new_traces = []
    for trace in traces:
        if len(trace) >= min_length:
            new_traces.append(trace)
    return new_traces

def find_connect_point(pre_point,cur_point,jitter_points,scs):
    sc_no = None
    sojourn = 0
    for x in jitter_points:
        if sc_no == None and scs.is_neighbor(pre_point[0],x[0]) and scs.is_neighbor(cur_point[0],x[0]) and pre_point[0] != x[0] and cur_point[0] != x[0]:
            sc_no = x[0]
        sojourn += x[1]
    if sc_no != None:
        if sojourn >= min_sojourn and sojourn <= max_sojourn:
            return (sc_no,sojourn)
        else:
            global sojourn_changes
            sojourn_changes += 1
            sojourn = int(referrence_mean_std[sc_no][0] + 2*(random.random()-0.5)*referrence_mean_std[sc_no][1])
            return (sc_no,sojourn)
    return None

def process_jitter_points(pre_point,cur_point,jitter_points,scs):
    trace = []
    trace.append(pre_point)
    if scs.is_neighbor(pre_point[0],cur_point[0]) and pre_point[0] != cur_point[0]:
        return trace
    else:
        connect_point = find_connect_point(pre_point,cur_point,jitter_points,scs)
        if connect_point != None:
            trace.append(connect_point)
            return trace
        else:
            trace.extend(jitter_points)
            return trace


def remove_jitter_points2(traces, scs):
    new_traces = []
    for trace in traces:
        new_trace = []
        pre_point = None
        cur_point = None
        jitter_points = []

        for i in range(len(trace)):
            if trace[i][1] < min_sojourn:
                if pre_point != None:
                    jitter_points.append(trace[i])
                continue
            elif pre_point == None:
                pre_point = trace[i]
                continue
            else:
                cur_point = trace[i]
                new_trace.extend(process_jitter_points(pre_point,cur_point,jitter_points,scs))
                pre_point = trace[i]
                jitter_points = []

        if cur_point != None:
            new_trace.append(cur_point)

        new_traces.append(new_trace)
    return new_traces

def split_traces(traces):
    new_traces = []
    for trace in traces:
        pre_index = None
        for i in range(len(trace)):
            if trace[i][1] >= min_sojourn:
                if pre_index == None:
                    pre_index = i
                    continue
                else:
                    continue
            else:
                if pre_index == None:
                    continue
                else:
                    new_traces.append(trace[pre_index:i])
                    pre_index = None
        if pre_index != None:
            new_traces.append(trace[pre_index:])
    return new_traces

# smooth the points with sojourn time larger than max_sojourn
def smooth_sojourn_time(traces):
    new_traces = []
    for trace in traces:
        new_trace = []
        for i in range(0,len(trace)):
            if trace[i][1] <= max_sojourn:
                new_trace.append(trace[i])
            else:
                sojourn = int(referrence_mean_std[trace[i][0]][0] + 2*(random.random()-0.5)*referrence_mean_std[ trace[i][0]][1])
                if sojourn < min_sojourn:
                    sojourn = max_sojourn - min_sojourn + int(random.random()*min_sojourn)
                new_trace.append((trace[i][0], sojourn))
        new_traces.append(new_trace)
    return new_traces

if __name__ == '__main__':
    f = open('traces1','rb')
    traces = pickle.load(f)
    f.close()
    f = open('scs', 'rb')
    scs = pickle.load(f)
    f.close()

    compute_referrence_mean_std(traces)

    traces = remove_short_traces(traces)
    traces_info(traces)

    traces = remove_jitter_points2(traces, scs)
    print '\n'
    print('sojourn_changes: %d' % sojourn_changes)
    print '\n'
    traces_info(traces)

    traces = split_traces(traces)
    traces = remove_short_traces(traces)
    traces = smooth_sojourn_time(traces)
    traces_info(traces)

    f = open('traces2', 'wb')
    pickle.dump(traces, f)
    f.close()
