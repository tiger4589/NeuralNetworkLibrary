using FeedForwardNeuralNetwork.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using SimpleNeuralNetwork.MnistReader;

namespace LibraryTest
{
    public static class FeedForwardTest
    {
        public static void FeedForwardNetworkTest()
        {
            TrainAgainstMNISTData();
        }

        private static void TrainAgainstMNISTData()
        {
            //We'll be using MNIST data to train our network to recognize digits from 0 to 9.

            //We know that MNIST Training data has 60,000 images.
            //We will be using the first 50,000 for training and the other 10,000 for the validation through our training.

            //First let's create our Neural Network.
            FeedForwardNeuralNet net = new FeedForwardNeuralNet(784, 100, 10, 0.1);

            TrainNetwork(net);
            TestNetwork(net);

            bool isFileSaved = net.SaveFile("mnist.txt");
            if (isFileSaved)
                Console.WriteLine("mnist.txt file has been successfully saved!");
            
            Console.ReadKey();
        }

        private static void TrainNetwork(FeedForwardNeuralNet net)
        {
            List<double[]> trainingInput = new List<double[]>();
            List<double[]> trainingOutput = new List<double[]>();
            List<double[]> validationInput = new List<double[]>();
            List<double[]> validationOutput = new List<double[]>();

            int i = 0;

            foreach (var image in MnistReader.ReadTrainingData())
            {
                double[] input = Encode(image.Data);
                double[] output = Encode(image.Label);

                if (i < 50000)
                {
                    trainingInput.Add(input);
                    trainingOutput.Add(output);
                }
                else
                {
                    validationInput.Add(input);
                    validationOutput.Add(output);
                }

                i++;
            }

            Console.WriteLine("Started Training:");
            net.Train(trainingInput,trainingOutput,30,10,5.0,validationInput,validationOutput);
            Console.WriteLine("Finished training.");
        }

        private static void TestNetwork(FeedForwardNeuralNet net)
        {
            Console.WriteLine("Started Testing");
            int testedImages = 0;
            int rightGuesses = 0;

            foreach (var image in MnistReader.ReadTestData())
            {
                double[] input = Encode(image.Data);
                double[] output = net.Run(input);

                int result = output.ToList().IndexOf(output.Max());

                if (result == image.Label)
                {
                    rightGuesses++;
                }

                testedImages++;
            }

            Console.WriteLine($"{testedImages} has been tested. {rightGuesses} were correctly classified.");
        }

        private static double[] Encode(byte output)
        {
            double[] result = new double[10];
            result[output] = 1.0;

            return result;
        }

        private static double[] Encode(byte[,] imageData)
        {
            double[] result = new double[imageData.Length];

            for (int j = 0; j < 28; j++)
            {
                for (int k = 0; k < 28; k++)
                {
                    result[j * 28 + k] = imageData[j, k] / 255.0;
                }
            }

            return result;
        }
    }
}