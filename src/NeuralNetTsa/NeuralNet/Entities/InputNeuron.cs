namespace NeuralNetTsa.NeuralNet.Entities;

public sealed class InputNeuron : NudgeNeuron<InputNeuron>
{
    public InputNeuron(int capacity) : base(capacity)
    {
    }

    public InputNeuron(double nudge, int capacity) : base(nudge, capacity)
    {
    }

    public override void Process()
    {
        foreach (PruneSynapse synapse in Outputs)
        {
            synapse.Signal = Inputs[0].Signal * synapse.Weight;
        }
    }
}