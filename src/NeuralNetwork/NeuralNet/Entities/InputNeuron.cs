
namespace NeuralNet.Entities
{
    public class InputNeuron : Neuron<InputNeuron>
    {
        public InputNeuron() : base()
        {
        }

        public InputNeuron(double nudge) : base()
        {
            Nudge = nudge;
        }

        public override void Process()
        {
            foreach (PruneSynapse synapse in Outputs)
                synapse.Signal = Inputs[0].Signal * synapse.Weight;
        }
    }
}
