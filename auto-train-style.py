import os
import sys
sys.path.append('utils/')
from eutils import *
from data_utils import *
from clean_utils import *

vers = ['amp', 'bbe', 'cjb', 'csb', 'darby', 'dra', 'erv', 'esv', 'kj21', 'mev', 'ncv', 'niv', 'nog', 'web', 'ylt']

for v1 in vers:
    for v2 in vers:
        d = json2load('configs/style.json')
        d['SRC_DICT'] = d['SRC_DICT'].replace('amp-bbe', '%s-%s'%(v1,v2))
        d['DST_DICT'] = d['DST_DICT'].replace('amp-bbe', '%s-%s'%(v1,v2))
        d['TRAIN_INPUT'] = d['TRAIN_INPUT'].replace('amp-bbe', '%s-%s'%(v1,v2))
        d['TRAIN_OUTPUT'] = d['TRAIN_OUTPUT'].replace('amp-bbe', '%s-%s'%(v1,v2))
        d['DEV_INPUT'] = d['DEV_INPUT'].replace('amp-bbe', '%s-%s'%(v1,v2))
        d['DEV_OUTPUT'] = d['DEV_OUTPUT'].replace('amp-bbe', '%s-%s'%(v1,v2))
        print d
        save2json(d, 'configs/auto.json')
        os.system('source activate tf12p27; export CUDA_VISIBLE_DEVICES=0; mkdir auto-train-style/%s-%s; python train.py -s auto-train-style/%s-%s -l auto-train-style/%s-%s -c auto'%(v1,v2, v1,v2, v1,v2))
        print('VERSION-%s-%s Finished!!!'%(v1,v2))
