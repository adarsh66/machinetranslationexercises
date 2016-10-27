#!/env python
import optparse
import sys
import models
import copy
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", default=0, type = "int", help="Verbose mode (default=0)")
optparser.add_option("-g", "--greedy", dest="greedy_iters", default=100, type = "int", help="Greedy decode iterations)")
opts = optparser.parse_args()[0]

#Hypothesis obj
hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, start_i, \
                        end_i, bitmap, fphrase, ephrase, swapped")

# Utility functions copied from compute-model-score script
def bitmap(sequence):
  """ Generate a coverage bitmap for a sequence of indexes """
  return reduce(lambda x,y: x|y, map(lambda i: long('1'+'0'*i,2), sequence), 0)

def bitmap2str(b, n, on='o', off='.'):
  """ Generate a length-n string representation of bitmap b """
  return '' if n==0 else (on if b&1==1 else off) + bitmap2str(b>>1, n-1, on, off)

def count_bitmap(b, n):
  """ Counts the number of bitmaps occupied """
  return 0 if n==0 else (1 if b&1==1 else 0) + count_bitmap(b>>1, n-1)

def extract_english(h, prev_swapped): 
  """A new version of this to handle swapped phrases.
     If we encounter a swapped phrase, we manually handle it by printing both phrases in right order
     And then we skip the phrase printing on the next hypothesis we encounter
  """
  if h.predecessor is None:
    return ""
  elif prev_swapped: #Skip printing phrase here
    return "%s" % extract_english(h.predecessor, h.swapped)
  elif h.swapped is True: #print it in the right order if swapped
    return "%s%s %s " % (extract_english(h.predecessor, h.swapped), h.ephrase.english, h.predecessor.ephrase.english)
  else:
    return "%s%s "  % (extract_english(h.predecessor, h.swapped), h.ephrase.english)

def maybe_write(s, verbosity):
  if opts.verbose >= verbosity:
    sys.stdout.write(s)
    sys.stdout.write('\n')
    sys.stdout.flush()

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french,())):
  if (word,) not in tm:
    tm[(word,)] = [models.phrase(word, 0.0)]

def get_score(h, ps1, ps2=None):
  lm_prob = tm_prob = 0.0
  tm_prob = h.logprob + ps1.logprob
  lm_state = h.lm_state
  for word in ps1.english.split():
    (lm_state, word_logprob) = lm.score(lm_state, word)
    lm_prob += word_logprob
  if ps2:
    tm_prob += ps2.logprob
    for word in ps2.english.split():
      (lm_state, word_logprob) = lm.score(lm_state, word)
      lm_prob += word_logprob
  return (lm_prob + tm_prob, lm_state)

def hyp_to_phrases(h):
  result = []
  def get_phrase(h, prev_swapped, ps):
    if h.predecessor is None or prev_swapped:
      return
    elif h.swapped:
      ps.append((h.predecessor.fphrase, h.ephrase.english))
      ps.append((h.fphrase, h.predecessor.ephrase))
      get_phrase(h.predecessor, h.swapped, ps)
    else:
      ps.append((h.fphrase, h.ephrase))
      get_phrase(h.predecessor, h.swapped, ps)
  get_phrase(h, False, result)
  return result

def replace(h):
  """ For each foreign phrase, it selects all the alternate english phrases """
  replaces = []
  if h.fphrase in tm:
    all_translations = tm[h.fphrase]
    for ts in all_translations:
      if h.ephrase != ts:
        logprob, lm_state = get_score(h.predecessor, ts)
        new_hypothesis = hypothesis(logprob, lm_state, h.predecessor, h.start_i, \
                      h.end_i,h.bitmap, h.fphrase, ts, h.swapped )
        replaces.append(new_hypothesis)
  return replaces

def 


def neighbourhood(h):
  return swap(h) + replce(h)

def greedy_decoder(seed, source):
  current = seed
  for i in range(greedy_iters):
    s_current = seed.logprob
    s = s_current
    for h in neighbourhood(seed):
      c = get_score(h.predecessor, h.ephrase)
      if c> s:
        s = c
        best = h
      if s == s_current:
        return current
      else:
        current = best
  return current


sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
  initial_hypothesis = hypothesis(0.0, lm.begin(), None, 0,0, bitmap(range(0)),  None, None, True)
  stacks = [{} for _ in f] + [{}]
  stacks[0][lm.begin()] = initial_hypothesis
  for i, stack in enumerate(stacks[:-1]):
    for h in sorted(stack.itervalues(), key=lambda h: -h.logprob)[:opts.s]: # prune
      for j in xrange(i+1,len(f)+1):
        if f[i:j] in tm:
          for phrase in tm[f[i:j]]:
            logprob, lm_state = get_score(h, phrase)
            logprob += lm.end(lm_state) if j == len(f) else 0.0
            new_v = bitmap(range(i, j)) | h.bitmap
            new_hypothesis = hypothesis(logprob, lm_state, h, i, j, new_v, f[i:j], phrase, False)
            v_count = count_bitmap(new_v, len(f))
            if new_v not in stacks[v_count] or stacks[v_count][new_v].logprob <logprob:
              stacks[v_count][new_v] = new_hypothesis 
              maybe_write( bitmap2str(new_v, len(f)), 2)
              maybe_write("#1 " + extract_english(new_hypothesis, False), 2)
              
            #we perform the phrase swap if the last french word ends on the current start
            #and the last hyp was not swapped as we can only swap adjacent phrases.
            if h.end_i == i and h.swapped is False:
              logprob, lm_state = get_score(h.predecessor, phrase, h.ephrase)
              logprob += lm.end(lm_state) if j == len(f) else 0.0
              new_v = bitmap(range(i,j)) | h.bitmap
              new_hypothesis = hypothesis(logprob, lm_state, h, h.start_i, j, \
                                new_v, f[h.start_i:j], phrase, True)
              v_count = count_bitmap(new_v, len(f))
              if new_v not in stacks[v_count] or stacks[v_count][new_v].logprob <logprob:
                stacks[v_count][new_v] = new_hypothesis 
                maybe_write ("#2 " + extract_english( new_hypothesis, False), 2)

  winner_stage1 = max(stacks[-1].itervalues(), key=lambda h: h.logprob)
  maybe_write (extract_english(winner_stage1, False), 0)

  #Implementing a greedy decoder with Swap & Replace methods implemented
  #Greedy decoding works by taking the seed sentence from the above decoder, and incrementaly improving it
  seed = hyp_to_phrase(winner_stage1)
  winner_stage2 = greedy_decode(winner_stage1, f)

  if opts.verbose:
    def extract_tm_logprob(h):
      return 0.0 if h.predecessor is None else h.ephrase.logprob + extract_tm_logprob(h.predecessor)
    tm_logprob = extract_tm_logprob(winner)
    sys.stderr.write("LM = %f, TM = %f, Total = %f\n" % 
      (winner.logprob - tm_logprob, tm_logprob, winner.logprob))



