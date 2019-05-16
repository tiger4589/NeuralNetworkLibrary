using System;
using FeedForwardNeuralNetwork.Utility;

namespace FeedForwardNeuralNetwork.Core
{
    internal class Weight
    {
        public double Value { get; set; }

        public Weight(Random random, int numberOfConnections = 0)
        {
            if (random == null)
            {
                throw new ArgumentNullException(nameof(random), $"{nameof(random)} is not initialized");
            }

            if (numberOfConnections < 1)
            {
                throw new ArgumentException($"{nameof(numberOfConnections)} can't be negative", nameof(numberOfConnections));
            }

            Value = Utilities.GetRandomValueBetweenMinAndMax(random) / Math.Sqrt(numberOfConnections);
        }

        public Weight(double weight)
        {
            Value = weight;
        }
    }
}