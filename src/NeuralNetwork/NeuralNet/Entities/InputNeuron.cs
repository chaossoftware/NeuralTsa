
namespace NeuralNet.Entities
{
    public class InputNeuron: Neuron
    {
        public InputNeuron(int outputsCount)
        {
            Outputs = new NewSynapse[outputsCount];
            Memory = new double[outputsCount];
            Best = new double[outputsCount];
            Input = 0;
        }

        public InputNeuron(int outputsCount, double nudge)
        {
            Outputs = new NewSynapse[outputsCount];
            Memory = new double[outputsCount];
            Best = new double[outputsCount];
            Nudge = nudge;
            Input = 0;
        }

        public override void Process()
        {
            foreach (NewSynapse synapse in Outputs)
                synapse.Signal = Input * synapse.Weight;
        }
    }
}
