The file lstm.py contains my implementation of LSTM network used to train on KDDcup dataset.

Usage:

	python lstm.py <epochs> <blocks> <cells> <peephole>

Example:
	
	python lstm.py 5 2 2 y

Note:
In general the network takes about 15 mins per epoch to train. 
The last argument 'peephole' can be 'y' for yes and 'n' for no.
The datasets must be in the same directory as lstm.py.
