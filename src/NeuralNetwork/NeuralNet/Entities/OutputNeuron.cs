
using System.Collections.Generic;

namespace NeuralNet.Entities
{
    public class OutputNeuron : Neuron
    {
        public NewSynapse BiasInput;

        public OutputNeuron(int inputsCount)
        {
            Outputs = new NewSynapse[1];
            Memory = new double[1];
            Best = new double[1];
            Inputs = new NewSynapse[inputsCount];
        }

        public OutputNeuron(int inputsCount, double nudge)
        {
            Outputs = new NewSynapse[1];
            Memory = new double[1];
            Best = new double[1];
            Inputs = new NewSynapse[inputsCount];
            Nudge = nudge;
        }

        public override void Process()
        {
            double arg = BiasInput.Weight;
            foreach (NewSynapse synapse in Inputs)
                arg += synapse.Signal;

            Outputs[0].Signal = arg;
        }
    }
}
