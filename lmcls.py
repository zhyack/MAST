#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
sys.path.append('utils/')
from eutils import *
from data_utils import *
from lm_utils import *
import numpy as np
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

plm1 = 'data/regina/stycls/kreutzer1.lm'
plm2 = 'data/regina/stycls/kreutzer2.lm'
plm3 = 'data/regina/stycls/bovary3.lm'
plm4 = 'data/regina/stycls/bovary2.lm'
pin = 'data/regina/stycls/test.in'
pout = 'test.out'

lm1 = initLM(plm1)
lm2 = initLM(plm2)
# lm1, lm2 = clipLM(lm1, lm2)

# lm3 = initLM(plm3)
# lm4 = initLM(plm4)
# lm3, lm4 = clipLM(lm3, lm4)
# lm3 = lm2

lines = codecs.open(pin, 'r', 'UTF-8').readlines()
f = codecs.open(pout, 'w', 'UTF-8')

for line in lines:
    l = line.split()
    lm_scores = []
    for n in range(10):
        score = 0.0
        if len(l)<=n+1:
            lm_scores.append(0.0)
            continue
        for i in range(len(l)-n):
            ng = ' '.join(l[i:i+n+1])
            s1 = getP(ng, lm1)*7
            s2 = getP(ng, lm2)*7
            ss = softmax(np.array([s1,s2])).tolist()
            # print(n, ng, s1,s2, ss)
            score += ss[0]-ss[1]
        lm_scores.append(score/(len(l)-n))
    final_score = 0.0
    for n in range(4):
        final_score += lm_scores[n]*(4-n)#*math.log((n+2)*2)
    print(lm_scores, final_score)
    if final_score >= 0:
        f.write('0\n')
    else:
        f.write('1\n')
f.close()
