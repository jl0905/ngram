# GenAI NGrams (assignment 1) Jeffery Lin

####

def readMethods():
	line_number = 1
	word_counts = {}
	methods = []
	with open("methods.txt", "r") as f:
		for line in f:
			if line_number == 15000: #COMMENT OUT 
				break
			if line_number % 2 == 0:
				#IF THE LINE IS A METHOD WE NEED TO DO STUFF
				methods.append(line.strip())
				line_number += 1
				continue

			line = line[6:].strip()

			if line in word_counts:
				word_counts[line] += 1
			else:
				word_counts[line] = 1
			line_number += 1
	return methods
####

class NGram:
	def __init__(self, N, tokenized_input):
		
		self.n = N
		self.ngram = self.constructNGram(tokenized_input, N)
		self.backoff_ngrams = [self.constructNGram(tokenized_input, previousN) for previousN in range(N - 1, 0, -1)]

		vocab = set()
		for entry in tokenized_input:
			for token in entry:
				vocab.add(token)
		self.vocab = list(vocab)

	def constructNGram(self, tokenized_input, n):
		'''Returns an ngram where the keys are all tokens n-1 and before and values are
			all nth tokens that follow that context. We represent the nth tokens through
			using nested dictionaries that are key value pairs of the nth token and its
			frequency in the training corpus
		
			training corpus is represented by list of tokenized methods and n hyperparameter'''
		if n == 1:
			ngram = {}
			for method in tokenized_input:
				for token in method.split(" "):
					if token in ngram.keys():
						ngram[token] += 1
					else:
						ngram[token] = 1
			return ngram
		ngram = {} # keys are tuple and value is dict
		for method in tokenized_input:
			tokens = ["<START>" for _ in range(n-1)]
			tokens.extend(method.split(" "))
			tokens.append("<END>")
			if tokens > self.n:
				pass
			#print(tokens)
			for i in range(len(tokens) - n + 1):
				#print(tokens[i:i+n-1])
				k = tuple(tokens[i:i+n-1])
				v = tokens[i+n-1]
				#print("key: ",k, " then the value: ", v)
				if k in ngram.keys():
					if v in ngram[k].keys():
						ngram[k][v] += 1
					else:
						ngram[k][v] = 1
				else:
					ngram[k] = {}
					ngram[k][v] = 1
		return ngram
		#...

	def getProbabilityOfToken(self, context, token):
		'''precondition is context has to be a tuple (w_i-n ... w_i - 1) and token is w_i'''
		if (self.n == 1):
			return self.ngram[token] / sum(self.ngram.values())
		if context in self.ngram.keys() and token in self.ngram[context].keys():
			return self.ngram[context][token] / sum(self.ngram[context].values())
		else:
			penalty = 1
			for i in range(len(self.backoff_ngrams)):#ngram in self.backoff_ngrams:
				ngram = self.backoff_ngrams[i]
				if (i == len(self.backoff_ngrams) - 1): # if n is one we have different key value pairs
					return penalty * ngram[token] / sum(ngram.values())
				if context in ngram.keys() and token in ngram[context].keys():
					return penalty * (ngram[context][token] / sum(ngram[context].values()))
				else:
					penalty = penalty * 0.8

	#def evaluate(self, )


# def pruneNGram(ngram, n):
# 	to_delete = []
# 	for context in ngram.keys():
# 		for potential_values in ngram[context].keys():
# 			if ngram[context][potential_values] < n:
# 				to_delete.append(tuple([context, potential_values]))
# 				#del ngram[context][potential_values]
# 	for pair in to_delete:
# 		del ngram[pair[0]][pair[1]]
# 	return ngram

	# def smoothingHelper(self, smoothed_list):
	# 	ret = []
	# 	for gram in smoothed_list:
	# 		for word in self.vocab:
	# 			new_gram = [thing for thing in gram]
	# 			new_gram.append(word)
	# 			ret.append(new_gram)
	# 	return ret


	# def smoothNGram(self):
	# 	smoothed_list = [word for word in self.vocab]
	# 	for _ in range(self.n):
	# 		smoothed_list = self.smoothingHelper(smoothed_list)
	# 	for gram in smoothed_list:
	# 		key = tuple(gram[:-1])
	# 		value = gram[-1]
	# 		if key not in self.ngram.keys():
	# 			self.ngram[key] = dict()
	# 		if value not in self.ngram[key].keys():
	# 			self.ngram[value] = 1
	# 		else:
	# 			self.ngram[value] += 1


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

quit()

N = 3
all_methods = readMethods()

test = NGram(3, all_methods)
#test.smoothNGram()
print(test.ngram[('arr', '.')])

def autoRegress(ngram):
	starting_context = ["<START>" for _ in range(N - 1)]
		#print(starting_context)

	for i in range(15):
		starting_context.append(generateNextToken(ngram.ngram, starting_context, N))
		if starting_context[-1] == "<END>": break
		#print(starting_context)

	print(starting_context)

autoRegress(test)

def calculatePerplexity():
	** (1/n)

def writeResults(filename):
	with open(filename+".json", "w") as f:
		tab = "	"
		def quote(s):
			return "\"" + s + "\""
		f.write("{\n")
		f.write(tab + "\"testSet\": \"provided.txt\",\n")
		f.write(tab + "\"perplexity\": 4.39,\n")
		f.write(tab + "\"data\": [\n")
		f.write(tab + tab + "{\n")
		f.write(tab + tab + tab + "\"index\": \"ID1\",\n")
		f.write(tab + tab + tab + "\"tokenizedCode\": \"public void run ( ) { }\",\n")
		f.write(tab + tab + tab + "\"contextWindow\": 3,\n")
		f.write(tab + tab + tab + "\"predictions\": [\n")
		f.write(tab + tab + tab + tab + "{\n")
		f.write(tab + tab + tab + tab + tab + "\"context\": [\"public\", \"void\"],\n")
		f.write(tab + tab + tab + tab + tab + "\"predToken\": \"run\",\n")
		f.write(tab + tab + tab + tab + tab + "\"predProbability\": 0.72,\n")
		f.write(tab + tab + tab + tab + tab + "\"groundTruth\": \"run\"\n")
		f.write(tab + tab + tab + tab + "},\n")
		f.write(tab + tab + tab + tab + "{\n")
		f.write(tab + tab + tab + tab + tab + "\"context\": [\"void\", \"run\"],\n")
		f.write(tab + tab + tab + tab + tab + "\"predToken\": \"(\",\n")
		f.write(tab + tab + tab + tab + tab + "\"predProbability\": 0.85,\n")
		f.write(tab + tab + tab + tab + tab + "\"groundTruth\": \"(\"\n")
		f.write(tab + tab + tab + tab + "}\n")
		f.write(tab + tab + tab + "]\n")
		f.write(tab + tab + "}\n")
		f.write(tab + "]\n")
		f.write("}\n")