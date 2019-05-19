using System;
using System.Collections.Generic;
using System.Linq;
using HopfieldNetwork.Utility;

namespace HopfieldNetwork.Core
{
    public class HopfieldNeuralNetwork
    {
        internal Neuron[] Neurons;
        internal Connection[] Connections;

        private double[] _inputs;

        public HopfieldNeuralNetwork(int numberOfNeurons)
        {
            Neurons = new Neuron[numberOfNeurons];

            for (int i = 0; i < numberOfNeurons; i++)
            {
                Neurons[i] = new Neuron();
            }

            int numberOfConnections = Utilities.Factorial(numberOfNeurons) / Utilities.Factorial(numberOfNeurons - 2);
            Connections = new Connection[numberOfConnections];

            for (int i = 0; i < numberOfConnections; i++)
            {
                Connections[i] = new Connection
                {
                    Weight = 0
                };
            }

            int conn = 0;
            for (int i = 0; i < numberOfNeurons; i++)
            {
                for (int j = 0; j < numberOfNeurons; j++)
                {
                    if (i == j)
                    {
                        continue;
                    }

                    Connections[conn++].SetNeurons(Neurons[i], Neurons[j]);
                }
            }
        }

        public double[] Run(double[] input)
        {
            double[] output = new double[input.Length];
            SetInput(Utilities.ConvertFromBinaryToBipolar(input));

            for (int i = 0; i < Neurons.Length; i++)
            {
                Connection[] connections = Connections.Where(x => x.FromNeuron == Neurons[i]).ToArray();
                int c = 0;
                double sum = 0;
                for (int j = 0; j < _inputs.Length; j++)
                {
                    if (i == j)
                    {
                        continue;
                    }

                    sum += _inputs[j] * connections[c++].Weight;
                }

                output[i] = sum;
            }

            return Utilities.GetBinaryOutput(output);
        }

        public void Train(double[] pattern)
        {
            double[] weights = new double[Connections.Length];

            pattern = Utilities.ConvertFromBinaryToBipolar(pattern);

            int index = 0;
            for (int i = 0; i < pattern.Length; i++)
            {
                for (int j = 0; j < pattern.Length; j++)
                {
                    if (i == j)
                    {
                        continue;
                    }

                    weights[index++] = pattern[i] * pattern[j];
                }
            }

            AddWeightsToConnections(weights);
        }

        private void AddWeightsToConnections(double[] weights)
        {
            for (int i = 0; i < Connections.Length; i++)
            {
                Connections[i].Weight += weights[i];
            }
        }

        private void SetInput(double[] inputs)
        {
            if (inputs.Length != Neurons.Length)
            {
                throw new ArgumentOutOfRangeException($"Neurons count: \"{Neurons.Length}\" is different than input count: \"{inputs.Length}\"");
            }

            _inputs = inputs;
        }
    }
}