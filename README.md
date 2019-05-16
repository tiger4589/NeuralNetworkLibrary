# NeuralNetworkLibrary
C# Machine Learning library - Will be updated as long as I am learning and experimenting with AI 

First version, which only contains a feed forward neural network has been published on Nuget.

---

To Install:

>Install-Package NeuralNetwork.Library -Version 1.0.0

---

**How to use?**

You can check the [library test sample code](https://github.com/tiger4589/NeuralNetworkLibrary/blob/master/LibraryTest/FeedForwardTest.cs) to check how to use the network. 

Also, here's a brief explanation:

First of all, you need to create your network by providing input layer count, hidden layer count, output layer count and your learning rate, such as:

    FeedForwardNeuralNet net = new FeedForwardNeuralNet(784, 100, 10, 0.1);
    
This will create a network with 784 neurons in the input layer, 100 neurons in hidden layer, 10 neurons in the output layer with the learning rate being equal to 0.1.

After preparing your inputs and corresponding outputs, you can call the `Train` method.

There's currently two train methods, one that is a standard back propagation and the other uses SGD learning algorithm.

The sample code is learning the network to classify the MNIST digits.

Here's my output after I ran it:

	Started Training:
	Epoch 1: 9310 / 10000
	Epoch 2: 9518 / 10000
	Epoch 3: 9574 / 10000
	Epoch 4: 9619 / 10000
	Epoch 5: 9664 / 10000
	Epoch 6: 9683 / 10000
	Epoch 7: 9681 / 10000
	Epoch 8: 9702 / 10000
	Epoch 9: 9709 / 10000
	Epoch 10: 9719 / 10000
	Epoch 11: 9724 / 10000
	Epoch 12: 9734 / 10000
	Epoch 13: 9733 / 10000
	Epoch 14: 9739 / 10000
	Epoch 15: 9752 / 10000
	Epoch 16: 9741 / 10000
	Epoch 17: 9757 / 10000
	Epoch 18: 9757 / 10000
	Epoch 19: 9756 / 10000
	Epoch 20: 9770 / 10000
	Epoch 21: 9760 / 10000
	Epoch 22: 9771 / 10000
	Epoch 23: 9762 / 10000
	Epoch 24: 9759 / 10000
	Epoch 25: 9767 / 10000
	Epoch 26: 9765 / 10000
	Epoch 27: 9774 / 10000
	Epoch 28: 9771 / 10000
	Epoch 29: 9770 / 10000
	Epoch 30: 9767 / 10000
	Finished training.
	Started Testing
	10000 has been tested. 9763 were correctly classified.
	File has been successfully saved!

It achieved a total of 97.63% success rate after 30 epochs of training. It could achieve more by slightly changing the hyper parameters or by training for more epochs.

