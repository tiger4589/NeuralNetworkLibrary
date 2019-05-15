using System;
using FeedForwardNeuralNetwork.Utility;

namespace FeedForwardNeuralNetwork.Core
{
    internal class Neuron
    {
        public double Bias { get; set; }
        public double Delta { get; set; }
        public double Value { get; set; }

        public Weight[] Weights { get; set; }

        public Neuron(int numberOfConnections, Random random)
        {
            Weights = new Weight[numberOfConnections];
            InitializeWeightsAndBias(random);
        }

        public Neuron(double bias, double delta, double value, double[] weightValues)
        {
            Bias = bias;
            Delta = delta;
            Value = value;

            Weights = new Weight[weightValues.Length];
            for (int i = 0; i < weightValues.Length; i++)
            {
                Weights[i] = new Weight(weightValues[i]);
            }
        }

        private void InitializeWeightsAndBias(Random random)
        {
            Bias = Utilities.GetRandomValueBetweenMinAndMax(random);

            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = new Weight(random, Weights.Length);
            }
        }
    }
}