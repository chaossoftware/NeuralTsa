using System;

namespace NeuralNet.Entities
{
    public class BiasNeuron : NudgeNeuron<BiasNeuron>
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
            throw new NotSupportedException("Bias neuron has no inputs, so not able to process something");
        }
    }
}
