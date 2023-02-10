using System;
using NeuralNetTsa.NeuralNet.Entities;
using ChaosSoft.Core;
using SciML.NeuralNetwork.Activation;

namespace NeuralNetTsa.NeuralNet.Activation;

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

public class PolynomialSixOrder : ComplexActivationFunction
{
    public override string Name => "Polynomial (6 order)";

    public PolynomialSixOrder() : base()
    {
    }

    public override double Phi(double arg) => 
        Neuron.Outputs[0].Weight + 
            arg * (Neuron.Outputs[1].Weight + 
                arg * (Neuron.Outputs[2].Weight + 
                    arg * (Neuron.Outputs[3].Weight + 
                        arg * (Neuron.Outputs[4].Weight + 
                            arg * (Neuron.Outputs[5].Weight + 
                                arg * Neuron.Outputs[6].Weight)))));

    public override double Dphi(double arg) =>
        Neuron.Outputs[1].Weight + 
            arg * (2d * Neuron.Outputs[2].Weight + 
                arg * (3d * Neuron.Outputs[3].Weight + 
                    arg * (4d * Neuron.Outputs[4].Weight + 
                        arg * (5d * Neuron.Outputs[5].Weight + 
                            arg * 6d * Neuron.Outputs[6].Weight))));
}

public class Rational : ComplexActivationFunction
{
    public override string Name => "Rational";

    public Rational() : base()
    {
    }

    public override double Phi(double arg) =>
        (Neuron.Outputs[0].Weight + 
            arg * (Neuron.Outputs[1].Weight + 
                arg * (Neuron.Outputs[2].Weight + 
                    arg * Neuron.Outputs[3].Weight))) 
            / (1d + 
            arg * (Neuron.Outputs[4].Weight + 
                arg * (Neuron.Outputs[5].Weight + 
                    arg * Neuron.Outputs[6].Weight)));

    public override double Dphi(double arg)
    {
        double f = Neuron.Outputs[0].Weight + 
            arg * (Neuron.Outputs[1].Weight + 
                arg * (Neuron.Outputs[2].Weight + 
                    arg * Neuron.Outputs[3].Weight));

        double df = Neuron.Outputs[1].Weight + 
            arg * (2d * Neuron.Outputs[2].Weight + 
                arg * 3d * Neuron.Outputs[3].Weight);

        double g = 1d + arg * (Neuron.Outputs[4].Weight + 
            arg * (Neuron.Outputs[5].Weight + 
                arg * Neuron.Outputs[6].Weight));

        double dg = Neuron.Outputs[4].Weight + 
            arg * (2d * Neuron.Outputs[5].Weight + 
                arg * 3d * Neuron.Outputs[6].Weight);

        return (g * df - f * dg) / (g * g);
    }
}

public class Special : ComplexActivationFunction
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
                Neuron.Outputs[3].Weight * FastMath.Pow2(ActivationFunctionsMath.Sech(arg));
}
