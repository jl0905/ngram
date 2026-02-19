# from collections import defaultdict
import random
# read

all_methods = []

word_counts = {}
line_number = 1
def readMethods():
	with open("methods.txt", "r") as f:
		for line in f:
			
			if line_number % 2 == 0:
				#IF THE LINE IS A METHOD WE NEED TO DO STUFF
				all_methods.append(line.strip())
				line_number += 1
				continue

			line = line[6:].strip()

			if line in word_counts:
				word_counts[line] += 1
			else:
				word_counts[line] = 1
			line_number += 1


def constructNGram(methodset, n):
	ngram = {} # keys are tuple and value is defaultdict
	for method in methodset:
		#print(["<START>" for _ in range(n-1)])
		tokens = ["<START>" for _ in range(n-1)]
		#tokens = method.split(" ")
		tokens.extend(method.split(" "))
		tokens.append("<END>")
		print(tokens)
		for i in range(len(tokens) - n + 1):
			#print(tokens[i:i+n-1])
			k = tuple(tokens[i:i+n-1])
			v = tokens[i+n-1]
			print("key: ",k, " then the value: ", v)
			if k in ngram.keys():
				if v in ngram[k].keys():
					ngram[k][v] += 1
				else:
					ngram[k][v] = 0
			else:
				ngram[k] = {}
	#...

def generateNextToken(ngram, context, n):
	curr = tuple(context[len(context)- n + 1:])
	distribution = ngram[curr]
	#distribution is like {"public": 0, "static": 100, "the": 3}
	
	# Convert counts to probabilities
	total_count = sum(distribution.values())
	if total_count == 0:
		return None
	
	# Create weighted random choice
	tokens = list(distribution.keys())
	weights = list(distribution.values())
	
	# Normalize weights to probabilities
	probabilities = [w / total_count for w in weights]
	
	# Randomly sample based on probabilities
	import random
	selected_token = random.choices(tokens, weights=probabilities, k=1)[0]
	
	return selected_token[0] if isinstance(selected_token, tuple) else selected_token
	

			
					

#print(all_methods[0])
print(random.choices(["one", "two", "three"], weights=[0.9, 0.09, 0.01], k=100))
constructNGram(all_methods[0:1], 3)
