
To install simply download the repo source files.
Needs python >3.6 and you can simply run ngram.py

accepts one custom file to evaluate on in the format: (you need to put it in the same directory as ngram.py)

python ./ngram.py filename.ext

Then you will be prompted, (simply type what you want.)

What do you want the output to be named? (include file extension): 

outputs are written in the directory

only hyperparameter that was tuned is N, the backoff smoothing penalty is 0.8 chosen arbitrarily.
