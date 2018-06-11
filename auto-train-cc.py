import os
import sys
sys.path.append('utils/')
from eutils import *
from data_utils import *
from clean_utils import *

# vers = ['BBE', 'DRA', 'DARBY', 'WEB', 'YLT', 'NIV', 'KJ21', 'AMP', 'CSB', 'CJB', 'ERV', 'ESV', 'NCV', 'MEV', 'NOG']
vers = ['BBE', 'DRA', 'DARBY', 'WEB', 'YLT']

for ver in vers:
    ver = ver.lower()
    d = json2load('configs/asv-web-cc.json')
    for k in d:
        try:
            d[k] = d[k].replace('asv-web', 'asv-%s'%(ver))
        except Exception:
            pass
    print d
    save2json(d, 'configs/auto-cc.json')
    os.system('source activate tf12p27; export CUDA_VISIBLE_DEVICES=5; mkdir auto-train-cc/%s; python ftrain.py -s auto-train-cc/%s -l auto-train-cc/%s -c auto-cc'%(ver, ver, ver))
    print('VERSION-%s Finished!!!'%(ver))
