using System;
using System.Collections.Generic;

namespace FeedForwardNeuralNetwork.Core
{
    internal class Layer
    {
        public Neuron[] Neurons { get; set; }

        public Layer(int numberOfNeurons, int numberOfConnections, Random random)
        {
            Neurons = new Neuron[numberOfNeurons];

            for (int i = 0; i < numberOfNeurons; i++)
            {
                Neurons[i] = new Neuron(numberOfConnections, random);
            }
        }

        public Layer(double[] biases, double[] deltas, double[] values, List<double[]> weights)
        {
            Neurons = new Neuron[biases.Length];

            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(biases[i], deltas[i], values[i], weights[i]);
            }
        }
    }
}