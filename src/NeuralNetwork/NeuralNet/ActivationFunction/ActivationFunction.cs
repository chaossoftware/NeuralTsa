using NeuralNet.Entities;
using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    public abstract class ActivationFunction
    {
        public abstract string Name { get; }

        public InputNeuron Neuron { get; set; }

        public bool AdditionalNeuron { get; set; } = false;

        public abstract double Phi(double arg);

        public abstract double Dphi(double arg);

        //Returns hyperbolic secant of arg
        protected double Sech(double arg) =>
            Math.Abs(arg) < 22d ?
            2d / (Math.Exp(arg) + Math.Exp(-arg)) :
            0d;

        protected void InitNetworkLayer()
        {
            Neuron = new InputNeuron();
            Neuron.Outputs = new List<PruneSynapse>(7);

            for (int i = 0; i < 7; i++)
                Neuron.Outputs[i] = new PruneSynapse(i, i);
        }
    }
}
