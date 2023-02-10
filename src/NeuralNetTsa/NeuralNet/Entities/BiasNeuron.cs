using System;

namespace NeuralNetTsa.NeuralNet.Entities;

public sealed class BiasNeuron : NudgeNeuron<BiasNeuron>
{
    public BiasNeuron(int capacity) : base(capacity)
    {
    }

    public BiasNeuron(double nudge, int capacity) : base(nudge, capacity)
    {
    }

    public override void Process() => 
        throw new NotSupportedException("Bias neuron has no inputs, so not able to process something");
}
