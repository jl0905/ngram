ABSTRACT

An N-Gram model is essentially a table of all possible permutations of N tokens (in other words all sentences, or grams) that are observed in the training set for the model. To utilize the NGram model, we access the conditional probability of the n-th token based on the previous n-1 tokens from our table. For the task of creating an NGram model, we attain a large mass of open source Java methods and we process these methods as training data for our model and also add back-off smoothing. The model is then evaluated twice; on a predetermined test set as well as a random test set from the training data. Evaluation is calculated through the perplexity via model performance on ground truth tokens in the testing set.   

       

We started the semester by being introduced to how we can mine public GitHub repositories for data to train our models on. By using 