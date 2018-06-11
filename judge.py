#coding: UTF-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import codecs
import argparse
parser = argparse.ArgumentParser(
    description="Compute Accuracy and Cross Entropy.")
parser.add_argument(
    "-acc",
    dest="file_accuracy",
    type=str,
    default=None,
    help="File_1, for accuracy.")
parser.add_argument(
    "-ce",
    dest="file_ce",
    type=str,
    default=None,
    help="File_2, for cross entropy.")
parser.add_argument(
    "-gold",
    dest="file_gold",
    type=str,
    help="The file of Gold Standard.")
args = parser.parse_args()

if args.file_gold==None:
    raise Exception("Use -gold to point to Gold Standard file.")
if args.file_accuracy==None and args.file_ce==None:
    raise Exception("At least one output file to judge, use -acc or -ce")

def readFile(pf):
    return codecs.open(pf, 'r', 'UTF-8').readlines()

gold = readFile(args.file_gold)

if args.file_accuracy:
    acc = readFile(args.file_accuracy)
    if len(gold) != len(acc):
        raise Exception("File_1 and Gold are not paralled.")
    correct = 0
    total = len(acc)
    for i in range(total):
        if acc[i].strip().rstrip() == gold[i].strip().rstrip():
            correct += 1
    print('Accuracy:\t %.2f'%(float(correct)/total*100))

if args.file_ce:
    ce = readFile(args.file_ce)
    if len(gold) != len(ce):
        raise Exception("File_2 and Gold are not paralled.")
    s = 0
    total = len(ce)
    for i in range(total):
        if (float(ce[i])==1.0):
            ce[i]=0.9
        elif (float(ce[i])==0.0):
            ce[i]=0.1
        s += float(gold[i])*math.log(float(ce[i])) + (1.0-float(gold[i]))*math.log(1.0-float(ce[i]))
    print('Cross Entropy:\t  %.2f'%(-s/total))
