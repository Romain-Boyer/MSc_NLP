The goal of this code is to implement skip-gram with negative-sampling from scratch.



To fit the model from path_train (training data : 1 sentence by line), 
and save the model to path_model (.json) : 
	python main.3.py --text path_train --model path_model

To test the model on 2 words (on a tabulated tab-separated csv file (path_test, with one header line containing
the words under columns 'word1' and 'word2' respectively) :
	python main.3.py --test --text path_test --model path_model
