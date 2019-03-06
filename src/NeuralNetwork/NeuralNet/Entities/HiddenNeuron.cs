using System;
using System.Collections.Generic;
using DeepLearn.NeuralNetwork.Base;
using NeuralNetwork;

namespace NeuralNet.Entities
{
    public class HiddenNeuron : Neuron
    {
        public static ActivationFunction Function;
        public NewSynapse BiasInput;

        public HiddenNeuron(int inputsCount, int outputsCount)
        {
            Inputs = new NewSynapse[inputsCount];
            Outputs = new NewSynapse[outputsCount];
            Memory = new double[outputsCount];
            Best = new double[outputsCount];
        }

        public HiddenNeuron(int inputsCount, int outputsCount, double nudge)
        {
            Inputs = new NewSynapse[inputsCount];
            Outputs = new NewSynapse[outputsCount];
            Memory = new double[outputsCount];
            Best = new double[outputsCount];
            Nudge = nudge;
        }

        public override void Process()
        {
            double arg = BiasInput.Weight;// + Inputs.Select(;
            foreach (NewSynapse synapse in Inputs)
                arg += synapse.Signal;

            foreach (NewSynapse synapse in Outputs)
                synapse.Signal = synapse.Weight * Function.Phi(arg);
        }
    }
}
