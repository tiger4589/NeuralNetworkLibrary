using System.Collections.Generic;

namespace HopfieldNetwork.Utility
{
    internal static class Utilities
    {
        public static double[] ConvertFromBinaryToBipolar(double[] list)
        {
            double[] bipolar = new double[list.Length];

            for (int i = 0; i < list.Length; i++)
            {
                bipolar[i] = 2 * list[i] - 1;
            }

            return bipolar;
        }

        public static List<double> ConvertFromBipolarToBinary(List<double> list)
        {
            List<double> binary = new List<double>();

            list.ForEach(x =>
            {
                binary.Add((x + 1) / 2);
            });

            return binary;
        }

        public static int Factorial(int i)
        {
            if (i <= 1)
            {
                return 1;
            }

            return i * Factorial(i - 1);
        }

        public static double[] GetBinaryOutput(double[] output)
        {
            double[] result = new double[output.Length];

            for (int i = 0; i < output.Length; i++)
            {
                result[i] = output[i] <= 0 ? 0 : 1;
            }

            return result;
        }
    }
}