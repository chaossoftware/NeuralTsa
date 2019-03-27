using System;
using System.Collections.Generic;
using NewMind.NeuralNet.Activation;
using NeuralAnalyser.NeuralNet.Entities;

namespace NeuralAnalyser.NeuralNet.Activation
{
    public abstract class ComplexActivationFunction : ActivationFunction
    {
        public InputNeuron Neuron { get; set; }

        public bool AdditionalNeuron { get; set; } = false;

        protected void InitNetworkLayer()
        {
            Neuron = new InputNeuron();
            Neuron.Outputs = new List<PruneSynapse>(7);

            for (int i = 0; i < 7; i++)
                Neuron.Outputs[i] = new PruneSynapse(i, i);
        }
    }

    public class PolynomialSixOrderFunction : ComplexActivationFunction
    {
        public override string Name => "Polynomial (6 order)";

        public PolynomialSixOrderFunction()
        {
            AdditionalNeuron = true;
            InitNetworkLayer();
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

    public class RationalFunction : ComplexActivationFunction
    {
        public override string Name => "Rational";

        public RationalFunction()
        {
            AdditionalNeuron = true;
            InitNetworkLayer();
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

    public class SpecialFunction : ComplexActivationFunction
    {
        public override string Name => "Special";

        public SpecialFunction()
        {
            AdditionalNeuron = true;
            InitNetworkLayer();
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
                    Neuron.Outputs[3].Weight * Math.Pow(Sech(arg), 2);
    }
}
