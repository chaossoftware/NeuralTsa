using NeuralAnalyser.NeuralNet.Activation;

namespace NeuralAnalyser.NeuralNet.Entities
{
    public class HiddenNeuron : NudgeNeuron<HiddenNeuron>
    {
        public static ActivationFunction Function;
        public PruneSynapse BiasInput;

        public HiddenNeuron() : base()
        {
        }

        public HiddenNeuron(double nudge) : base()
        {
            Nudge = nudge;
        }

        public override void Process()
        {
            double arg = BiasInput.Weight;
            foreach (PruneSynapse synapse in Inputs)
                arg += synapse.Signal;

            foreach (PruneSynapse synapse in Outputs)
                synapse.Signal = synapse.Weight * Function.Phi(arg);
        }
    }
}
