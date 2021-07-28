# neural_network_mark1
Neural Network with one hidden layer to classify arbitrarily separated points
numpy used. more abstract libraries not used.
data stored in json format

Execution order:
(1) Supply the data using a program of data_gen family (uncomment the plotting portions to see the nature of the data), or use your own data
(2) Train the neural network using the data with trainer.py. Change the number of iterations niter as per your requirement.
(3) Use plotcost.py to see if the cost function is declining.
(4) Plot the test data using plotdata.py parallet to the training set to see the performance of the neural network.
(5) Use hypothesis() from hypothesis.py for the classification application.
