#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict
from decimal import Decimal as D

optparser = optparse.OptionParser()
optparser.add_option("-b", "--bitext", dest="bitext", default="data/small.txt", help="Parallel corpus (default data/dev-test-train.de-en)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
(opts, _) = optparser.parse_args()

sys.stderr.write('Training with IBM Model 1...')
bitext = [[sentence.strip().split() for sentence in pair.split(' ||| ')] for pair in open(opts.bitext)][:opts.num_sents]
n = 0
#e_keys = set()
#for (f, e) in bitext:
#    for e in set(e):
#        e_keys.add(e)
# Initialized to a uniform probability
#t = defaultdict(lambda: D(1.0/len(e_keys)))
t = defaultdict(lambda: D(1.0/len(bitext)))

sys.stderr.write("Iterative EM method to begin\n")
#IBM Model 1
for i in range(5):
    count_fe = defaultdict(D)
    count_e = defaultdict(D)
    z_norm = defaultdict(D)
    for (n, (f, e)) in enumerate(bitext):
        for fi in set(f):
            z_norm[fi] = D()
            #normalization
            for ej in set(e):
                z_norm[fi] += t[(fi, ej)]
            for ej in set(e):
                count_fe[(fi, ej)] += t[(fi, ej)] / z_norm[fi]
                count_e[ej] += t[(fi, ej)] / z_norm[fi]
        if n % 1000 == 0:
            sys.stderr.write(".")
    sys.stderr.write('now we estimate the probability \n')
    for (fi, ej) in count_fe.keys():
        t[(fi, ej)] = count_fe[(fi, ej)] / count_e[ej]

#Decoder of the model
for (f, e) in bitext:
    for (i, fi) in enumerate(f):
        best_prob = 0
        best_j = 0
        for (j, ej) in enumerate(e):
            if t[(fi, ej)] > best_prob:
                best_prob = t[(fi, ej)]
                best_j = j
        sys.stdout.write("%i-%i " % (i,best_j))
    sys.stdout.write("\n")