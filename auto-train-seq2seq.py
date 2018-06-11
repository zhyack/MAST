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
    d = json2load('configs/asv-bbe.json')
    d['DATASET']='asv-%s'%(ver)
    d['SRC_DICT'] = d['SRC_DICT'].replace('asv-ylt', 'asv-%s'%(ver))
    d['DST_DICT'] = d['DST_DICT'].replace('asv-ylt', 'asv-%s'%(ver))
    d['TRAIN_INPUT'] = d['TRAIN_INPUT'].replace('asv-ylt', 'asv-%s'%(ver))
    d['TRAIN_OUTPUT'] = d['TRAIN_OUTPUT'].replace('asv-ylt', 'asv-%s'%(ver))
    d['DEV_INPUT'] = d['DEV_INPUT'].replace('asv-ylt', 'asv-%s'%(ver))
    d['DEV_OUTPUT'] = d['DEV_OUTPUT'].replace('asv-ylt', 'asv-%s'%(ver))
    print d
    save2json(d, 'configs/auto.json')
    os.system('source activate tf12p27; export CUDA_VISIBLE_DEVICES=0; mkdir auto-train-seq2seq/%s; python train.py -s auto-train-seq2seq/%s -l auto-train-seq2seq/%s -c auto'%(ver, ver, ver))
    print('VERSION-%s Finished!!!'%(ver))
