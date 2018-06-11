import os
import sys
sys.path.append('utils/')
from eutils import *
from data_utils import *
from clean_utils import *

# vers = ['web', 'cjb', 'nog', 'amp', 'bbe', 'cjb', 'csb', 'darby', 'dra', 'erv', 'esv', 'kj21', 'mev', 'ncv', 'niv', 'nog', 'web', 'ylt']
vers = ['bbe', 'dra', 'darby', 'web', 'ylt']
nb = {'bbe': ['web', 'darby'], 'dra':['web', 'darby'], 'darby': ['web', 'dra'], 'web': ['dra', 'darby'], 'ylt':['web', 'darby']}

# nb = {'bbe':['kj21','esv','mev','web',''], 'dra':['kj21','darby','esv','web',''],\
#  'darby':['kj21','web','dra','esv',''], 'web':['esv','kj21','mev','nog',''],\
#  'ylt':['kj21','darby','mev','web',''], 'amp':['web','dra','darby','ylt',''],\
#  'cjb':['web','dra','csb','ylt',''], 'csb':['web','niv','mev','kj21',''],\
#  'erv':['web','niv','kj21','darby',''], 'esv':['web','mev','kj21','darby',''],\
#  'kj21':['darby','web','dra','ylt',''], 'mev':['esv','web','kj21','niv',''], 'niv':['web','csb','esv','mev',''],\
#  'nog':['web','dra','niv','csb',''], 'ncv':['web','erv','csb','darby','']}
#
# nb = {'bbe':['csb','esv','kj21','mev','niv'], 'dra':['darby','esv','niv','nog','amp'],\
#  'darby':['dra','ylt','esv','kj21','mev'], 'web':['csb','esv','mev','nog','niv'],\
#  'ylt':['amp','erv','kj21','mev','ncv'], 'amp':['dra','web','ylt','mev','esv'],\
#  'cjb':['dra','ylt','csb','niv','web'], 'csb':['esv','mev','niv','ncv','cjb'],\
#  'erv':['ylt','niv','nog','ncv','dra'], 'esv':['dra','csb','mev','niv','web'],\
#  'kj21':['dra','darby','ylt','mev','web'], 'mev':['dra','csb','esv','niv','web'], 'niv':['csb','esv','mev','ncv','nog'],\
#  'nog':['dra','web','ylt','mev','bbe'], 'ncv':['ylt','csb','erv','niv','nog']}

for ver in vers:
    d = json2load('configs/asv-bbe-ma.json')
    for k in d:
        try:
            d[k] = d[k].replace('asv-bbe', 'asv-%s'%(ver))
        except Exception:
            pass
    d['MODEL_PREFIX']=['%s'%(ver)]+['%s'%(s) for s in nb[ver]]
    print d
    save2json(d, 'configs/auto.json')
    os.system('export CUDA_VISIBLE_DEVICES=3; mkdir auto-train-mast/%s; python mtrain.py -s auto-train-mast/%s -l auto-train-mast/%s -c auto'%(ver, ver, ver))
    print('VERSION-%s Finished!!!'%(ver))
