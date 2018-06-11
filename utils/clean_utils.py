import os
import random
import codecs
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

def printDictInfo(d):
    wc = 0
    for v in d.values():
        wc += v
    print('%d --- %d/%d --- %.2f%%'%(len(d), d['<UNK>'], wc, (1-float(d['<UNK>'])/wc)*100))

def reclean(d,pf):
    f = codecs.open(pf, 'r', 'UTF-8')
    lines = f.readlines()
    f.close()
    f = codecs.open(pf, 'w', 'UTF-8')
    for l in lines:
        l = l.split()
        for w in l:
            if w not in d:
                f.write('<UNK> ')
            else:
                f.write(w + ' ')
        f.write('\n')
    f.close()

def getsublist(l,s):
    ret = []
    for i in s:
        ret.append(l[i])
    return ret

def denoise(lines):
    ret = []
    for l in lines:
        l = l.split()
        t = []
        for i in range(len(l)):
            w = l[i]
            r = random.random()
            if r<0.1:
                pass
            elif r<0.2:
                t.append(w)
                t.append(w)
            elif r<0.3:
                if i+1!=len(l):
                    t.append(l[i+1])
                    l[i+1] = w
            else:
                t.append(w)
        ret.append(' '.join(t)+'\n')
    return ret

def all_tokenize_by_word(lines):
    for i in range(len(lines)):
        lines[i] = ' '.join(word_tokenize(lines[i]))
    return lines
def sent_tokenize_by_word(s):
    return ' '.join(word_tokenize(s))
def para_tokenize_by_sent(s):
    return sent_tokenize(s)
