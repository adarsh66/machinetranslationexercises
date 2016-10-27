#! /bin/sh


for pe in 0 0.5 1 2; do
  for pef in 0 0.5 1 2; do
    for pfe in 0 0.5 1 2; do
	python2.7 rerank -w 'p(e)='$pe' p(e|f)='$pef' p_lex(f|e)='$pfe'' | python2.7 compute-bleu
    done
  done
done
