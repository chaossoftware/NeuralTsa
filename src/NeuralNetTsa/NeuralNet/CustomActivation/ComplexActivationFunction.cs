using ChaosSoft.NeuralNetwork.Activation;
using NeuralNetTsa.NeuralNet.Entities;

namespace NeuralNetTsa.NeuralNet.CustomActivation;

public abstract class ComplexActivationFunction : IActivationFunction
{
    private const int Capacity = 7;

    protected ComplexActivationFunction()
    {
        AdditionalNeuron = true;

        Neuron = new InputNeuron(Capacity);

        for (int i = 0; i < Capacity; i++)
        {
            Neuron.Outputs.Add(new PruneSynapse(i, i));
        }
    }

    public InputNeuron Neuron { get; }

    public bool AdditionalNeuron { get; }

    public abstract string Name { get; }

    public abstract double Dphi(double arg);

    public abstract double Phi(double arg);
}
