using System;

namespace FeedForwardNeuralNetwork.Utility
{
    internal static class Utilities
    {
        private const double MIN_VALUE = -1;
        private const double MAX_VALUE = 1;

        public static double GetRandomValueBetweenMinAndMax(Random random)
        {
            return random.NextDouble() * (MAX_VALUE - MIN_VALUE) + MIN_VALUE;
        }
    }
}