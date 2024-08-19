"""
File: interactive.py
Name: Jane
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""
import util
import submission


def main():
	weights = {}
	for line in open('weights', 'rb'):
		line = line.decode('latin-1')
		s, w = line.split('\t', 1)
		weights[s] = float(w)

	util.interactivePrompt(submission.extractWordFeatures, weights)


if __name__ == '__main__':
	main()
