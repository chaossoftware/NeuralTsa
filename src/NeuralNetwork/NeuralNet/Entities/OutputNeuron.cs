namespace NeuralAnalyser.NeuralNet.Entities
{
    public class OutputNeuron : NudgeNeuron<OutputNeuron>
    {
        public PruneSynapse BiasInput;

        public OutputNeuron() : base()
        {
        }

        public OutputNeuron(double nudge) : base()
        {
            Nudge = nudge;
        }

        public override void Process()
        {
            double arg = BiasInput.Weight;
            foreach (PruneSynapse synapse in Inputs)
                arg += synapse.Signal;

            Outputs[0].Signal = arg;
        }
    }
}
