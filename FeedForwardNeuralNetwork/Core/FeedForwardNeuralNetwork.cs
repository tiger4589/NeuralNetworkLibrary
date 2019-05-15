using FeedForwardNeuralNetwork.Utility;
using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

namespace FeedForwardNeuralNetwork.Core
{
    public class FeedForwardNeuralNetwork
    {
        internal Layer InputLayer;
        internal Layer HiddenLayer;
        internal Layer OutputLayer;
        internal double LearningRate;

        /// <summary>
        /// Create a new neural network by giving number of neurons in each layer and the learning rate.
        /// </summary>
        /// <param name="inputLayerNeuronsNumber">Number of neurons in input layer</param>
        /// <param name="hiddenLayerNeuronsNumber">Number of neurons in hidden layer</param>
        /// <param name="outputLayerNeuronsNumber">Number of neurons in output layer</param>
        /// <param name="learningRate">Neural Network Learning Rate</param>
        public FeedForwardNeuralNetwork(int inputLayerNeuronsNumber, int hiddenLayerNeuronsNumber,
                                        int outputLayerNeuronsNumber, double learningRate)
        {

            if (inputLayerNeuronsNumber < 1)
            {
                throw new ArgumentException("Input Layer neurons number can't be less than 1",nameof(inputLayerNeuronsNumber));
            }

            if (hiddenLayerNeuronsNumber < 1)
            {
                throw new ArgumentException("Hidden Layer neurons number can't be less than 1", nameof(hiddenLayerNeuronsNumber));
            }

            if (outputLayerNeuronsNumber < 1)
            {
                throw new ArgumentException("Output Layer neurons number can't be less than 1", nameof(outputLayerNeuronsNumber));
            }

            Random random = new Random();

            InputLayer = new Layer(inputLayerNeuronsNumber, 0, random);
            HiddenLayer = new Layer(hiddenLayerNeuronsNumber, inputLayerNeuronsNumber, random);
            OutputLayer = new Layer(outputLayerNeuronsNumber, hiddenLayerNeuronsNumber, random);
            LearningRate = learningRate;
        }

        /// <summary>
        /// This constructor will only be used when deserializing the network from the saved file.
        /// </summary>
        /// <param name="inputLayer">Input Layer read from file</param>
        /// <param name="hiddenLayer">Hidden Layer read from file</param>
        /// <param name="outputLayer">Output layer read from file</param>
        /// <param name="learningRate">Learning Rate read from file</param>
        internal FeedForwardNeuralNetwork(Layer inputLayer, Layer hiddenLayer, Layer outputLayer, double learningRate)
        {
            InputLayer = inputLayer;
            HiddenLayer = hiddenLayer;
            OutputLayer = outputLayer;
            LearningRate = learningRate;
        }

        /// <summary>
        /// Run the network against the given input and return the output
        /// </summary>
        /// <param name="input">Given input to run</param>
        /// <returns>Returns the output layer values in a double array</returns>
        public double[] Run(double[] input)
        {
            if (input == null)
            {
                throw new ArgumentNullException(nameof(input), "Input is null");
            }

            if (input.Length == 0)
            {
                throw new ArgumentException("Input layer can't have 0 values", nameof(input));
            }

            if (input.Length != InputLayer.Neurons.Length)
            {
                throw new ArgumentException("Input values are not equal to the network input layer neurons count", nameof(input));
            }

            for (int i = 0; i < InputLayer.Neurons.Length; i++)
            {
                InputLayer.Neurons[i].Value = input[i];
            }

            //From Input To Hidden
            int hiddenLayerNeurons = HiddenLayer.Neurons.Length;
            int inputLayerNeurons = InputLayer.Neurons.Length;

            for (int i = 0; i < hiddenLayerNeurons; i++)
            {
                double value = 0.0;
                for (int j = 0; j < inputLayerNeurons; j++)
                {
                    value += HiddenLayer.Neurons[i].Weights[j].Value * InputLayer.Neurons[j].Value;
                }

                HiddenLayer.Neurons[i].Value = Sigmoid(value + HiddenLayer.Neurons[i].Bias);
            }

            //From Hidden to Output
            int outputLayerNeurons = OutputLayer.Neurons.Length;
            for (int i = 0; i < outputLayerNeurons; i++)
            {
                double value = 0.0;
                for (int j = 0; j < hiddenLayerNeurons; j++)
                {
                    value += OutputLayer.Neurons[i].Weights[j].Value * HiddenLayer.Neurons[j].Value;
                }

                OutputLayer.Neurons[i].Value = Sigmoid(value + OutputLayer.Neurons[i].Bias);
            }

            //Get the values of the last layer (output) and return it
            double[] output = new double[outputLayerNeurons];
            for (int i = 0; i < outputLayerNeurons; i++)
            {
                output[i] = OutputLayer.Neurons[i].Value;
            }

            return output;
        }

        /// <summary>
        /// Train the neural network to classify the given inputs
        /// </summary>
        /// <param name="listOfInputs">List of inputs from training set</param>
        /// <param name="listOfOutputs">List of outputs of the training set</param>
        /// <param name="numberOfEpochs">Number of epochs that the network should train</param>
        /// <param name="miniBatchSize">Since we're using the SGD learning algorithm, we need to provide the miniBatchSize</param>
        /// <param name="lambda">Provide the lambda for the SGD learning algorithm</param>
        /// <param name="testInputData">List of Input validation set</param>
        /// <param name="testOutputData">List of Output validation set</param>
        public void Train(List<double[]> listOfInputs, List<double[]> listOfOutputs, int numberOfEpochs = 30, int miniBatchSize = 10, double lambda = 5.0,
                          List<double[]> testInputData = null, List<double[]> testOutputData = null)
        {
            if (listOfInputs == null)
            {
                throw new ArgumentNullException(nameof(listOfInputs), "Input list can't be null");
            }

            if (listOfOutputs == null)
            {
                throw new ArgumentNullException(nameof(listOfOutputs), "Output list can't be null");
            }

            if (listOfInputs.Count == 0)
            {
                throw new ArgumentException("Input list count can't be equal to zero",nameof(listOfInputs));
            }

            if (listOfOutputs.Count == 0)
            {
                throw new ArgumentException("Output list count can't be equal to zero", nameof(listOfOutputs));
            }

            if (listOfInputs.Count != listOfOutputs.Count)
            {
                throw new InvalidOperationException($"{nameof(listOfInputs)} and {nameof(listOfOutputs)} doesn't have the same number of values");
            }

            List<Tuple<double[], double[]>> trainingList = new List<Tuple<double[], double[]>>();

            for (int i = 0; i < listOfInputs.Count; i++)
            {
                trainingList.Add(new Tuple<double[], double[]>(listOfInputs[i],listOfOutputs[i]));
            }

            Sgd(trainingList, numberOfEpochs, miniBatchSize, lambda, testInputData, testOutputData);
        }

        /// <summary>
        /// SGD Learning Algorithm
        /// </summary>
        /// <param name="inputAndOutputDataList">Training Set Lists</param>
        /// <param name="epochs">Number of epochs to train</param>
        /// <param name="miniBatchSize">Mini batch size</param>
        /// <param name="lambda">Lambda</param>
        /// <param name="testInputData">Validation Input Set</param>
        /// <param name="testOutputData">Validation Output Set</param>
        private void Sgd(List<Tuple<double[], double[]>> inputAndOutputDataList, int epochs, int miniBatchSize,
                         double lambda, List<double[]> testInputData = null,
                         List<double[]> testOutputData = null)
        {

            int trainingDataCount = inputAndOutputDataList.Count;

            for (int i = 0; i < epochs; i++)
            {
                inputAndOutputDataList.Shuffle();
                List<List<Tuple<double[], double[]>>> allMiniBatches = new List<List<Tuple<double[], double[]>>>();

                for (int k = 0; k < trainingDataCount; k = k + miniBatchSize)
                {
                    List<Tuple<double[], double[]>> miniBatch = new List<Tuple<double[], double[]>>();
                    for (int j = 0; j < miniBatchSize; j++)
                    {
                        miniBatch.Add(inputAndOutputDataList[j + k]);
                    }

                    allMiniBatches.Add(miniBatch);
                }

                foreach (List<Tuple<double[], double[]>> batch in allMiniBatches)
                {
                    UpdateMiniBatch(batch, lambda, trainingDataCount);
                }

                if (testInputData != null && testOutputData != null)
                {
                    int testDataCount = testInputData.Count;
                    int guessedNumber = Evaluate(testInputData, testOutputData);
                    Console.WriteLine($"Epoch {i + 1}: {guessedNumber} / {testDataCount}");
                }
                else
                {
                    Console.WriteLine($"Finished Epoch {i + 1}");
                }

                //BackupFile(((double)guessedNumber / (double)testDataCount) * 100.0);
            }
        }

        private void UpdateMiniBatch(List<Tuple<double[], double[]>> batch, double lambda, int trainingDataCount)
        {
            double[] outputNablaB = new double[OutputLayer.Neurons.Length];
            double[] hiddenNablaB = new double[HiddenLayer.Neurons.Length];

            double[][] outputNablaW = new double[OutputLayer.Neurons.Length][];
            for (int i = 0; i < outputNablaW.Length; i++)
            {
                outputNablaW[i] = new double[HiddenLayer.Neurons.Length];
            }
            double[][] hiddenNablaW = new double[HiddenLayer.Neurons.Length][];
            for (int i = 0; i < hiddenNablaW.Length; i++)
            {
                hiddenNablaW[i] = new double[InputLayer.Neurons.Length];
            }

            for (int i = 0; i < batch.Count; i++)
            {
                Tuple<double[], double[], double[][], double[][]> backProp = BackPropWithCrossEntropy(batch[i].Item1, batch[i].Item2);

                for (int j = 0; j < backProp.Item1.Length; j++)
                {
                    outputNablaB[j] += backProp.Item1[j];
                }

                for (int j = 0; j < backProp.Item2.Length; j++)
                {
                    hiddenNablaB[j] += backProp.Item2[j];
                }

                for (int j = 0; j < outputNablaW.Length; j++)
                {
                    for (int k = 0; k < outputNablaW[j].Length; k++)
                    {
                        outputNablaW[j][k] += backProp.Item3[j][k];
                    }
                }

                for (int j = 0; j < backProp.Item4.Length; j++)
                {
                    for (int k = 0; k < backProp.Item4[j].Length; k++)
                    {
                        hiddenNablaW[j][k] += backProp.Item4[j][k];
                    }
                }
            }

            for (int i = 0; i < OutputLayer.Neurons.Length; i++)
            {
                OutputLayer.Neurons[i].Bias -= (LearningRate / batch.Count) * outputNablaB[i];
                for (int j = 0; j < HiddenLayer.Neurons.Length; j++)
                {
                    OutputLayer.Neurons[i].Weights[j].Value = (1 - LearningRate * (lambda / trainingDataCount)) * OutputLayer.Neurons[i].Weights[j].Value -
                       (LearningRate / batch.Count) * outputNablaW[i][j];
                }
            }

            for (int i = 0; i < HiddenLayer.Neurons.Length; i++)
            {
                HiddenLayer.Neurons[i].Bias -= (LearningRate / batch.Count) * hiddenNablaB[i];
                for (int j = 0; j < InputLayer.Neurons.Length; j++)
                {
                    HiddenLayer.Neurons[i].Weights[j].Value = (1 - LearningRate * (lambda / trainingDataCount)) * HiddenLayer.Neurons[i].Weights[j].Value -
                                                                 (LearningRate / batch.Count) * hiddenNablaW[i][j];
                }
            }
        }

        private Tuple<double[], double[], double[][], double[][]> BackPropWithCrossEntropy(double[] input, double[] desiredOutput)
        {
            double[] outputNablaB = new double[OutputLayer.Neurons.Length];
            double[] hiddenNablaB = new double[HiddenLayer.Neurons.Length];

            double[][] outputNablaW = new double[OutputLayer.Neurons.Length][];
            for (int i = 0; i < outputNablaW.Length; i++)
            {
                outputNablaW[i] = new double[HiddenLayer.Neurons.Length];
            }
            double[][] hiddenNablaW = new double[HiddenLayer.Neurons.Length][];
            for (int i = 0; i < hiddenNablaW.Length; i++)
            {
                hiddenNablaW[i] = new double[InputLayer.Neurons.Length];
            }

            Run(input);

            for (int i = 0; i < OutputLayer.Neurons.Length; i++)
            {
                OutputLayer.Neurons[i].Delta = OutputLayer.Neurons[i].Value - desiredOutput[i];
                outputNablaB[i] = OutputLayer.Neurons[i].Delta;
            }

            for (int i = 0; i < outputNablaW.Length; i++)
            {
                for (int j = 0; j < outputNablaW[i].Length; j++)
                {
                    outputNablaW[i][j] = outputNablaB[i] * HiddenLayer.Neurons[j].Value;
                }
            }

            for (int i = 0; i < OutputLayer.Neurons.Length; i++)
            {
                for (int j = 0; j < HiddenLayer.Neurons.Length; j++)
                {
                    hiddenNablaB[j] += outputNablaB[i] * OutputLayer.Neurons[i].Weights[j].Value;
                }
            }
            for (int i = 0; i < HiddenLayer.Neurons.Length; i++)
            {
                hiddenNablaB[i] *= SigmoidDerivative(HiddenLayer.Neurons[i].Value);
            }

            for (int i = 0; i < hiddenNablaB.Length; i++)
            {
                for (int j = 0; j < InputLayer.Neurons.Length; j++)
                {
                    hiddenNablaW[i][j] += hiddenNablaB[i] * InputLayer.Neurons[j].Value;
                }
            }

            return new Tuple<double[], double[], double[][], double[][]>(outputNablaB, hiddenNablaB, outputNablaW,
                hiddenNablaW);

        }

        private int Evaluate(List<double[]> inputs, List<double[]> outputs)
        {
            int sum = 0;

            for (int i = 0; i < inputs.Count; i++)
            {
                sum += Evaluate(inputs[i], outputs[i]);
            }

            return sum;
        }

        private int Evaluate(double[] input, double[] output)
        {
            double[] currentOutput = Run(input);

            if (CheckResultWithOutput(currentOutput, output))
            {
                return 1;
            }

            return 0;
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            return x * (1 - x);
        }

        private bool CheckResultWithOutput(double[] result, double[] output)
        {
            double highest = double.NegativeInfinity;
            int bestIndex = -1;

            for (int i = 0; i < result.Length; i++)
            {
                if (result[i] > highest)
                {
                    highest = result[i];
                    bestIndex = i;
                }
            }

            for (int i = 0; i < output.Length; i++)
            {
                if (Math.Abs(output[i] - 1.0) < 0.0001)
                {
                    if (bestIndex == i)
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        /// <summary>
        /// Save the neural network in a json file to load it later
        /// </summary>
        /// <param name="fileName">file path and name</param>
        /// <returns>Returns true if the file was successfully saved</returns>
        public bool SaveFile(string fileName)
        {
            if (string.IsNullOrWhiteSpace(fileName))
            {
                throw new ArgumentNullException(nameof(fileName), "File Name is empty");
            }

            try
            {
                File.WriteAllText(fileName, Save());
                return true;
            }
            catch (Exception)
            {
                throw;
            }
        }

        private string Save()
        {
            return JsonConvert.SerializeObject(this, Formatting.Indented, new FeedForwardNeuralNetworkJsonConvertor(this));
        }

        /// <summary>
        /// Load the network from the specified file path and name
        /// </summary>
        /// <param name="fileName">File Path and Name</param>
        /// <returns>Returns the Network Object to use</returns>
        public static FeedForwardNeuralNetwork Load(string fileName)
        {
            string data;
            try
            {
                data = File.ReadAllText(fileName);
                if (string.IsNullOrWhiteSpace(data))
                {
                    throw new InvalidOperationException("File couldn't be read.");
                }
            }
            catch (Exception)
            {
                throw;
            }

            FeedForwardNeuralNetworkJsonConvertor convertor = new FeedForwardNeuralNetworkJsonConvertor(data);
            FeedForwardNeuralNetwork net = convertor.DeserializeData();
            return net;
        }
    }
}