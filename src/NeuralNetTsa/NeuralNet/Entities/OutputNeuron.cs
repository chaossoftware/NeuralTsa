namespace NeuralNetTsa.NeuralNet.Entities;

public sealed class OutputNeuron : NudgeNeuron<OutputNeuron>
{

    public OutputNeuron(int capacity) : base(capacity)
    {
    }

    public OutputNeuron(double nudge, int capacity) : base(nudge, capacity)
    {
    }

    public PruneSynapse BiasInput { get; set; }

    public override void Process()
    {
        double arg = BiasInput.Weight;

        foreach (PruneSynapse synapse in Inputs)
        {
            arg += synapse.Signal;
        }

        Outputs[0].Signal = arg;
    }
}
