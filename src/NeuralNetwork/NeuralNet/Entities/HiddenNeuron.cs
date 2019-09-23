using NewMind.NeuralNet.Activation;

namespace NeuralAnalyser.NeuralNet.Entities
{
    public class HiddenNeuron : NudgeNeuron<HiddenNeuron>
    {
        private ActivationFunction activationFunction;
        public PruneSynapse BiasInput;

        public HiddenNeuron() : base()
        {
        }

        public HiddenNeuron(double nudge) : base()
        {
            Nudge = nudge;
        }

        public HiddenNeuron(ActivationFunction activationFunction, double nudge)
        {
            this.activationFunction = activationFunction;
            this.Nudge = nudge;
        }

        public override void Process()
        {
            double arg = BiasInput.Weight;
            foreach (PruneSynapse synapse in Inputs)
                arg += synapse.Signal;

            foreach (PruneSynapse synapse in Outputs)
                synapse.Signal = synapse.Weight * activationFunction.Phi(arg);
        }
    }
}
