using ChaosSoft.NeuralNetwork.Activation;
using System;

namespace NeuralNetTsa.NeuralNet.CustomActivation;

public sealed class Special : ComplexActivationFunction
{
    public override string Name => "Special";

    public Special() : base()
    {
    }

    public override double Phi(double arg) =>
        Math.Abs(arg) < 22d ?

        Neuron.Outputs[0].Weight +
                arg * (Neuron.Outputs[1].Weight +
                    arg * Neuron.Outputs[2].Weight) +
                Neuron.Outputs[3].Weight *
                    (1d - 2d / (Math.Exp(2d * arg) + 1d)) :

        Neuron.Outputs[0].Weight +
                arg * (Neuron.Outputs[1].Weight +
                    arg * Neuron.Outputs[2].Weight) +
                        Neuron.Outputs[3].Weight * Math.Sign(arg);

    public override double Dphi(double arg) =>
        Neuron.Outputs[1].Weight +
            arg * 2d * Neuron.Outputs[2].Weight +
                Neuron.Outputs[3].Weight * Pow2(ActivationFunctionsMath.Sech(arg));

    private static double Pow2(double num) =>
        num * num;
}