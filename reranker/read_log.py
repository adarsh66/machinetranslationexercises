#!/usr/bin/env python
import sys

def main():
	for line in sys.stdin:
		if 'Weights:' in line:
			_, pef, pfe, pe = line.split()
			_,pef = pef.split('=')
			_,pfe = pfe.split('=')
			_,pe = pe.split('=')
		elif 'BLEU:' in line:
			_, bleu_score = line.split()
			echo_line(pef, pfe, pe, bleu_score)

def echo_line(pef, pfe, pe, bleu_score):
	sep= ','
	print pef + sep + pfe + sep + pe + sep + bleu_score

if __name__ == '__main__':
	main()

