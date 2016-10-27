There are three python programs here (`-h` for usage):

-`align.py` aligns words.

-`check-alignments.py` checks that the entire dataset is aligned, and
  that there are no out-of-bounds alignment points.

-`score-alignments.py` computes alignment error rate.

The commands work in a pipeline. For instance:

   > python2.7 align -t 0.9 -n 1000 | python2.7 check | python2.7 grade -n 5

The `data` directory contains:

-`dev-test-train.de-en` is a German-English parallel text 

-`dev.align` is a human alignment of the first 150 sentences of the
  parallel text (the _development_ section).
