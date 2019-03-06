using System;

namespace NeuralNet.Entities
{
    public class BiasNeuron : Neuron<BiasNeuron>
    {
        public BiasNeuron() : base()
        {
        }

        public BiasNeuron(double nudge)
        {
            Nudge = nudge;
        }

        public override void Process()
        {
            throw new Exception("Bias neuron has no inputs, so not able to process something");
        }
    }
}
