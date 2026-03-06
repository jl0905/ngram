# GenAI NGrams (assignment 1) Jeffery Lin

import json


def readMethods(filename, size):
	line_number = 0
	# word_counts = {}
	methods = []
	with open(filename, "r") as f:
		for line in f:
			if filename == "methods.txt":
				if line_number == size * 2: 
					break
				if line_number % 2 == 1:
					#IF THE LINE IS A METHOD WE NEED TO DO STUFF
					methods.append(line.strip().split(" "))
					line_number += 1
					continue
			else:
				if line_number == size:
					break
				methods.append(line.strip().split(" "))
			# line = line[6:].strip()

			# if line in word_counts:
			# 	word_counts[line] += 1
			# else:
			# 	word_counts[line] = 1
			line_number += 1
	return methods
####

class NGram:
	def __init__(self, N, tokenized_input):
		tokenized_input = self.preprocessUnknownTokens(tokenized_input)
		
		self.n = N
		self.ngram = self.constructNGram(tokenized_input, N)
		self.backoff_ngrams = [self.constructNGram(tokenized_input, previousN) for previousN in range(N - 1, 0, -1)]

		vocab = set()
		for method in tokenized_input:
			for token in method:
				vocab.add(token)
		self.vocab = list(vocab)

	def preprocessUnknownTokens(self, tokenized_input):
		temp_unigram = self.constructNGram(tokenized_input, 1)
		for i in range(len(tokenized_input)):
			method = tokenized_input[i]
			for j in range(len(method)):
				token = method[j]
				if temp_unigram[token] < 3: #IF LESS THAN 3 occurences we treat it as unknown
					method[j] = "<UNK>"
		return tokenized_input

	def constructNGram(self, tokenized_input, n):
		'''Returns an ngram where the keys are all tokens n-1 and before and values are
			all nth tokens that follow that context. We represent the nth tokens through
			using nested dictionaries that are key value pairs of the nth token and its
			frequency in the training corpus
		
			training corpus is represented by list of tokenized methods and n hyperparameter'''
		if n == 1:
			ngram = {}
			for method in tokenized_input:
				for token in method:#.split(" "):
					if token in ngram.keys():
						ngram[token] += 1
					else:
						ngram[token] = 1
			return ngram
		ngram = {} # keys are tuple and value is dict
		for method in tokenized_input:
			tokens = ["<START>" for _ in range(n-1)]
			tokens.extend(method)#.split(" "))
			tokens.append("<END>")
			# if tokens > self.n:
			# 	pass
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
		# if token not encountered in vocab its oov
		if token not in self.vocab:
			token = "<UNK>"
		if (self.n == 1):
			return self.ngram[token] / sum(self.ngram.values())
		if context in self.ngram.keys() and token in self.ngram[context].keys():
			return self.ngram[context][token] / sum(self.ngram[context].values())
		else:
			penalty = 1
			for i in range(len(self.backoff_ngrams)):#ngram in self.backoff_ngrams:
				ngram = self.backoff_ngrams[i]
				if (i == len(self.backoff_ngrams) - 1): # if n is one we have different key value pairs
					# if token not in ngram:
					# 	return 0
					return penalty * ngram[token] / sum(ngram.values())
				if context in ngram.keys() and token in ngram[context].keys():
					return penalty * (ngram[context][token] / sum(ngram[context].values()))
				else:
					penalty = penalty * 0.8

	def evaluate(self, evaluation_set, write_to=""):
		from decimal import Decimal, getcontext
		# Set precision high enough for our calculations
		getcontext().prec = 50
		
		perplexity = Decimal(1)  # Use Decimal for arbitrary precision
		probabilities = []
		n = self.n
		epsilon = Decimal('1e-10')  # Small number to prevent zero probabilities 
		for method in evaluation_set:
			tokens = ["<START>" for _ in range(self.n-1)]
			tokens.extend(method)#.split(" "))
			tokens.append("<END>")
			for i in range(len(tokens) - n + 1):
				k = tuple(tokens[i:i+n-1])
				v = tokens[i+n-1]
				pred_probability = self.getProbabilityOfToken(k, v)
				# Clamp probability to avoid zero
				pred_probability = max(Decimal(pred_probability), epsilon)
				probabilities.append(float(pred_probability))  # Store as float for JSON
				perplexity *= (Decimal(1) / pred_probability)
		# Convert back to float for final result
		perplexity = float(perplexity ** (Decimal(1) / Decimal(len(probabilities))))
		#print(f"Final perplexity: {perplexity}")
		if write_to != "":
			self.writeResults(write_to, evaluation_set, perplexity, probabilities)
		return perplexity

	def writeResults(self, filename, evaluation_set, perplexity, probabilities):
		tokenized_methods = evaluation_set
		token_i = 0
		ID = 1
		n = self.n
		with open(filename, "w") as f:
			tab = "	"
			f.write("{\n")
			f.write(tab + "\"testSet\": \"provided.txt\",\n")
			f.write(tab + "\"contextWindow\": " + str(self.n) + ",\n")
			f.write(tab + "\"perplexity\": " + str(perplexity) + ",\n")
			f.write(tab + "\"data\": [\n")
			#for tokenized_method in tokenized_methods:
			for j in range(len(tokenized_methods)):
				tokenized_method = tokenized_methods[j]
				f.write(tab + tab + "{\n")
				f.write(tab + tab + tab + "\"index\": \"" + "ID" + str(ID) + "\",\n")
				ID += 1
				f.write(tab + tab + tab + "\"tokenizedCode\": " + json.dumps("".join([mtd + " " for mtd in tokenized_method])[:-1]) + ",\n")
				f.write(tab + tab + tab + "\"predictions\": [\n")
				# f.write(tab + tab + tab + tab + "{\n")
				# f.write(tab + tab + tab + tab + tab + "\"context\": [\"public\", \"void\"],\n")
				# f.write(tab + tab + tab + tab + tab + "\"predToken\": \"run\",\n")
				# f.write(tab + tab + tab + tab + tab + "\"predProbability\": 0.72,\n")
				# f.write(tab + tab + tab + tab + tab + "\"groundTruth\": \"run\"\n")
				# f.write(tab + tab + tab + tab + "},\n")
				tokens = ["<START>" for _ in range(self.n-1)]
				tokens.extend(tokenized_method)#.split(" "))
				tokens.append("<END>")
				for i in range(len(tokens) - n + 1):
					k = tuple(tokens[i:i+n-1])
					v = tokens[i+n-1]
					f.write(tab + tab + tab + tab + "{\n")
					f.write(tab + tab + tab + tab + tab + "\"context\": " + json.dumps(list(k)) + ",\n")
					f.write(tab + tab + tab + tab + tab + "\"groundTruth\": " + json.dumps(v) + ",\n")
					f.write(tab + tab + tab + tab + tab + "\"predProbability\": " + str(probabilities[token_i]) + "\n")
					token_i += 1
					if (i == len(tokens) - n):
						f.write(tab + tab + tab + tab + "}\n")
					else:
						f.write(tab + tab + tab + tab + "},\n")
				f.write(tab + tab + tab + "]\n")
				if j == len(tokenized_methods) - 1:
					f.write(tab + tab + "}\n")
				else:
					f.write(tab + tab + "},\n")
			f.write(tab + "]\n")
			f.write("}\n")

all_methods = readMethods("methods.txt", 60000)
t1 = all_methods[:15000]
t2 = all_methods[:25000]
t3 = all_methods[:35000]
unseen = all_methods[35000:]
import random
random.seed(42)  # For reproducible results
unseen_sample = random.sample(unseen, 2000)  # Take 2000 random samples from unseen
validation = unseen_sample[:1000]  # First 1000 for validation
personal_test = unseen_sample[1000:]  # Next 1000 for testing

real_test = readMethods("test.txt", -1)

def benchmark():
	for i in [3, 5, 7]:
		for data in [t1, t2, t3]:
			model = NGram(i, data)
			print(f"The perplexity for a {i}-gram model trained from {len(data)} entries of data is: {model.evaluate(validation)}")
			del model
		
model = NGram(3, t2)

#print(test.ngram[tuple(["private"])])
#test.evaluate(test_methods, "results.json")
import sys

if len(sys.argv) > 1:
	fname = sys.argv[1]

	user_data = readMethods(fname, -1)

	output_fname = input("What do you want the output to be named? (include file extension): ")

	model.evaluate(user_data, output_fname)

xd = input("\nevaluatte preexisting test data? (Y/N): \n")

if 'y' in xd.lower():
	model.evaluate(real_test, "results.xxxxxx.json")
	model.evaluate(personal_test, "results.yyyyyy.json")
else:
	pass