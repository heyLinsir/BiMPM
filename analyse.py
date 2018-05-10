import os, sys, time

import numpy as np

keys = ['epoch', 'step', 'loss', 'distributionloss', 'stdloss', 'precise']

def parseLine(line):
    line = line.strip()
    action = None
    for i, c in enumerate(line):
        if c == '-':
            action = line[:i]
            if 'attention' in action:
            	action = action.split(' ')[1]
            description = line[i + 3:]
            break
    if action is None:
        return None, None
    description = description.replace(' ', '').split(',')
    description = [x.split(':') for x in description]
    result = {}
    for x in description:
        result[x[0]] = eval(x[1])
    return action, result

def get_max_precise(max_epoch=-1):
    r = open(sys.argv[1], 'r')
    result = []
    for line in r:
        action, parse_result = parseLine(line)
        if action is None:
            continue
        if action == 'train':
            continue
        if max_epoch >= 0 and parse_result['epoch'] > max_epoch:
            continue
        if parse_result['epoch'] >= len(result):
            result.append({'test': [], 'dev': []})
        result[parse_result['epoch']][action].append(parse_result['precise'])
    _result = []
    for x in result:
        for test, dev in zip(x['test'], x['dev']):
            _result.append((dev, test))
    _result = sorted(_result, key=lambda x: x[0], reverse=True)
    result = [{'test': np.max(x['test']), 'dev': np.max(x['dev'])} for x in result]
    for i, x in enumerate(result):
        print('epoch:%d\ttest:%f\tdev:%f' % (i, x['test'], x['dev']))
    test_result = np.max([x['test'] for x in result])
    dev_result = np.max([x['dev'] for x in result])
    print('max precise---test:%f\tdev:%f' % (test_result, dev_result))
    print('max dev is %f, test is %f at that time' % (_result[0][0], _result[0][1]))

if __name__ == '__main__':
    try:
        get_max_precise(eval(sys.argv[2]))
    except:
        get_max_precise()
