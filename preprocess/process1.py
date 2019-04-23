from SmallCells import SmallCells
import sys
from os import listdir
from os.path import isfile
import pickle

scs = SmallCells(40.02, 39.97, 116.35, 116.3, 1.0/300)

def get_duration(pre_t, t):
    return int((t-pre_t)*24*3600 + 0.5)

def get_records(f):
    for i in range(6):
        f.readline()

    pre_sc = None
    pre_t = None
    line = f.readline()
    records = []
    record = []
    while line:
        temp = line.split(',')
        lati = float(temp[0])
        longti = float(temp[1])
        t = float(temp[4])

        sc = scs.get_smallcell_no(lati, longti)
        if sc == None:
            line = f.readline()
            continue

        if pre_sc == None:
            pass
        else:
            if scs.is_neighbor(pre_sc, sc) and get_duration(pre_t, t) < 10:
                pass
            else:
                records.append(record)
                record = []
        record.append((sc,t))
        pre_sc = sc
        pre_t = t

        line = f.readline()

    records.append(record)
    return records

def get_traces(records):
    traces = []
    for record in records:
        if record == []:
            continue
        trace = []
        pre_sc = record[0][0]
        pre_t = record[0][1]
        for i in range(1,len(record)):
            if record[i][0] != pre_sc:
                sojourn = get_duration(pre_t, record[i][1])
                trace.append((pre_sc, sojourn))
                pre_sc = record[i][0]
                pre_t = record[i][1]
        if trace != []:
            traces.append(trace)

    return traces

def process_file(f):
    records = get_records(f)
    traces = get_traces(records)
    return traces

# process origin Data directory
def process_dir(dname):
    traces = []
    dirs = listdir(dname)
    for x in dirs:
        dirname = dname + '/' + x + '/Trajectory/'
        files = listdir(dirname)
        for y in files:
            f = open(dirname + y,'r')
            traces.extend(process_file(f))
    return traces

if __name__ == '__main__':
    traces = process_dir('../Data')
    f = open('traces1', 'wb')
    pickle.dump(traces, f)
    f.close()
    f = open('scs', 'wb')
    pickle.dump(scs, f)
    f.close()
