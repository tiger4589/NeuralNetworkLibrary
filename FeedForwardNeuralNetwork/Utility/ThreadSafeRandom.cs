using System;
using System.Threading;

namespace FeedForwardNeuralNetwork.Utility
{
    internal static class ThreadSafeRandom
    {
        [ThreadStatic] private static Random _local;

        public static Random ThisThreadsRandom
        {
            get
            {
                return _local ??
                       (_local =
                           new Random(unchecked (Environment.TickCount * 31 + Thread.CurrentThread.ManagedThreadId)));
            }
        }
    }
}