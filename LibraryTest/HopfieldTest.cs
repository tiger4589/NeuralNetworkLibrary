using System;
using System.Collections.Generic;
using System.Text;
using HopfieldNetwork.Core;

namespace LibraryTest
{
    public static class HopfieldTest
    {
        public static void HopefieldNeuralNetworkTest()
        {
            HopfieldNeuralNetwork network = new HopfieldNeuralNetwork(4);
            double[] inputToTrain = new double[] { 1, 1, 0, 0 };
            network.Train(inputToTrain);
            double[] firstInput = new double[] { 1, 1, 0, 0 };
            double[] firstOutput = network.Run(firstInput);
            double[] secondInput = new double[] { 1, 0, 0, 0 };
            double[] secondOutput = network.Run(secondInput);

            Console.WriteLine($"Trained network for {ConvertToTrueAndFalsePresentation(inputToTrain)}");
            Console.WriteLine($"Output for {ConvertToTrueAndFalsePresentation(firstInput)} is {ConvertToTrueAndFalsePresentation(firstOutput)}");
            Console.WriteLine($"Output for {ConvertToTrueAndFalsePresentation(secondInput)} is {ConvertToTrueAndFalsePresentation(secondOutput)}");

            Console.ReadKey();
        }

        private static string ConvertToTrueAndFalsePresentation(double[] list)
        {
            StringBuilder sb = new StringBuilder();

            sb.Append("[");

            for (int i = 0; i < list.Length; i++)
            {
                if (list[i] <= 0)
                {
                    sb.Append("F");
                }
                else
                {
                    sb.Append("T");
                }

                if (i != list.Length - 1)
                {
                    sb.Append(",");
                }
            }

            sb.Append("]");
            return sb.ToString();
        }
    }
}