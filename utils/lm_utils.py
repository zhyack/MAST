#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from eutils import *
from data_utils import *

import os
import random
import codecs

def initLM(pm):
    f = codecs.open(pm, 'r', 'UTF-8')
    lines = f.readlines()
    f.close()
    ret = dict()
    for l in lines:
        l = l.split()
        if len(l)<3 or l[0]=='ngram':
            continue
        try:
            float(l[-1])
        except ValueError:
            l.append('0.0')
        k = ' '.join(l[1:-1])
        if k not in ret:
            ret[k]=dict()
            ret[k]['p'] = float(l[0])
            ret[k]['bp'] = float(l[-1])

    return ret

def getBP(k, lm):
    if k in lm:
        return lm[k]['bp']
    else:
        return -1.0

def getP(k, lm):
    if k in lm:
        return lm[k]['p']
    else:
        return -10.0
        # if k.find(' ')==-1:
        #     return getBP(k, lm)
        # else:
        #     return getBP(k[:k.rfind(' ')], lm)+getP(k[k.find(' ')+1:], lm)


def clipLM(lm1, lm2):
    def isclose(x,y):
        thres = 1e-2
        if math.fabs(x-y)<=thres:
            return True
        else:
            return False
    ret1, ret2 = {}, {}
    ret1 = copy.deepcopy(lm1)
    ret2 = copy.deepcopy(lm2)
    for k in ret1.keys():
        if k in ret2 and isclose(ret1[k]['p'], ret2[k]['p']) and isclose(ret1[k]['bp'], ret2[k]['bp']):
            del(ret1[k])
            del[ret2[k]]
            # print(k)
    # for k in ret1.keys():
    #     if k not in ret2 and ret1[k]['p']>0 or k in ret2 and ret1[k]['p']-ret2[k]['p']>1.0:
    #         ret1[k]['p']=3.0
    # for k in ret2.keys():
    #     if k not in ret1 and ret2[k]['p']>0 or k in ret1 and ret2[k]['p']-ret1[k]['p']>1.0:
    #         ret2[k]['p']=3.0
    save2json(ret1, 'k1.json')
    save2json(ret2, 'k2.json')

    return ret1, ret2

def LMScore(outputs, plm, pdict):
    global lm, dict_dst
    if lm == None:
        lm, dict_dst = initLM(plm, pdict)
    max_len = outputs.shape[1]
    batch_size = len(outputs)
    ret = np.zeros([batch_size,max_len,len(dict_dst)], np.float32)
    for i in range(batch_size):
        predictions = outputs[i].argmax(axis=-1)
        for j in range(max_len):
            p = int(predictions[j])
            k = ' '.join([str(w) for w in predictions[max(0,j-2):j+1]])
            if j>1:
                reward=(tanh(getP(k)/5.0)+0.5)*2
                ret[i][j][p]+=reward
                # ret[i][j-1][int(predictions[j-1])]+=reward
                # ret[i][j-2][int(predictions[j-2])]+=reward
            # bk = ' '.join([str(w) for w in predictions[j:min(j+3,batch_size)]])
            # if j>1 and j<batch_size-2:
            #     ret[i][j][p]=((tanh(getP(k)/5.0)+0.5)/5.0 + (tanh(getP(bk)/5.0)+0.5)/5.0) / 2.0
            # elif j>1:
            #     ret[i][j][p]=(tanh(getP(k)/5.0)+0.5)/5.0
            # elif j<batch_size-2:
            #     ret[i][j][p]=(tanh(getP(bk)/5.0)+0.5)/5.0
            # if j>1:
            #     ret[i][j][p]=(tanh(getP(k)/5.0)+0.5)/10.0
        # for j in range(max_len):
        #     ret[i][j][int(predictions[j])]/=float(min(3,j+1))
        # print([ret[i][j][int(predictions[j])] for j in range(max_len)])
    return ret
