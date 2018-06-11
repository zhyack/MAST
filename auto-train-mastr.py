import os
import sys
sys.path.append('utils/')
from eutils import *
from data_utils import *
from clean_utils import *
import copy

vers = ['amp', 'bbe', 'cjb', 'csb', 'darby', 'dra', 'erv', 'esv', 'kj21', 'mev', 'ncv', 'niv', 'nog', 'web', 'ylt']

for ver in vers:
    d = json2load('configs/asv-bbe-ma.json')
    for k in d:
        try:
            d[k] = d[k].replace('asv-bbe', 'asv-%s'%(ver))
        except Exception:
            pass
    d['MODEL_PREFIX'] = ['asv-%s-02-'%(ver)]
    tv = copy.deepcopy(vers)
    random.shuffle(tv)
    pp = 0
    while(len(d['MODEL_PREFIX'])<5):
        if tv[pp]!=ver:
            d['MODEL_PREFIX'].append('asv-%s-02-'%(tv[pp]))
        pp += 1
    print d
    save2json(d, 'configs/auto.json')
    os.system('source activate tf12p27; export CUDA_VISIBLE_DEVICES=3; mkdir auto-train-mastr4/%s; python mtrain.py -s auto-train-mastr4/%s -l auto-train-mastr4/%s -c auto'%(ver, ver, ver))
    print('VERSION-%s Finished!!!'%(ver))
