namespace HopfieldNetwork.Core
{
    internal class Connection
    {
        public Neuron FromNeuron { get; set; }
        public Neuron ToNeuron { get; set; }
        public double Weight { get; set; }

        public void SetNeurons(Neuron from, Neuron to)
        {
            FromNeuron = from;
            ToNeuron = to;
        }
    }
}