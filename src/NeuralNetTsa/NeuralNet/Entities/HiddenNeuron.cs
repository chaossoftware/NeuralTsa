using ChaosSoft.NeuralNetwork.Activation;

namespace NeuralNetTsa.NeuralNet.Entities;

public sealed class HiddenNeuron : NudgeNeuron<HiddenNeuron>
{
    public static IActivationFunction Function;

    public HiddenNeuron(int capacity) : base(capacity)
    {
    }

    public HiddenNeuron(double nudge, int capacity) : base(nudge, capacity)
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

        double multiplier = Function.Phi(arg);

        foreach (PruneSynapse synapse in Outputs)
        {
            synapse.Signal = synapse.Weight * multiplier;
        }
    }
}
