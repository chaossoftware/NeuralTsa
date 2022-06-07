using NeuralAnalyser.NeuralNet;
using NeuralAnalyser.NeuralNet.Entities;
using System;
using System.Linq;

namespace NeuralAnalyser
{
    public class ConsoleNetVisualizer
    {
        private const string InactiveNeuron = "░░";
        private const string ActiveNeuron = "▓▓";
        private const string NeutralNeuron = "▒▒";
        private const int MinOffset = 1;

        public static int Visualize(SciNeuralNet neuralNet, int offset)
        {
            int inCount = neuralNet.Params.Dimensions;
            int hiddenCount = neuralNet.Params.Neurons;

            int maxLayerItemsCount = Math.Max(inCount, hiddenCount);
            int height = maxLayerItemsCount * 2;

            int inYOffset = maxLayerItemsCount / inCount + MinOffset;
            int hidYOffset = maxLayerItemsCount / hiddenCount + MinOffset;
            int outYOffset = maxLayerItemsCount / neuralNet.OutputLayer.Neurons.Length + MinOffset;

            double maxSynapseValue = neuralNet.Connections[0].Max(s => s.Signal);
            double minSynapseValue = neuralNet.Connections[0].Min(s => s.Signal);

            double oneThird = (maxSynapseValue - minSynapseValue) / 3d;
            double firstThird = minSynapseValue + oneThird;
            double secondThird = minSynapseValue + 2d * oneThird;

            int i = 1;

            foreach (InputNeuron inn in neuralNet.InputLayer.Neurons)
            {
                string synapseBrush = GetSynapseBrush(inn.Outputs[0].Signal, firstThird, secondThird);
                Console.SetCursorPosition(0, offset + i * inYOffset);
                Console.Write($"■ {hiddenCount} {synapseBrush}");
                i++;
            }

            maxSynapseValue = neuralNet.Connections[1].Max(s => s.Signal);
            minSynapseValue = neuralNet.Connections[1].Min(s => s.Signal);

            oneThird = (maxSynapseValue - minSynapseValue) / 3d;
            firstThird = minSynapseValue + oneThird;
            secondThird = minSynapseValue + 2d * oneThird;
            i = 1;

            foreach (HiddenNeuron hn in neuralNet.HiddenLayer.Neurons)
            {
                string synapseBrush = GetSynapseBrush(hn.Outputs[0].Signal, firstThird, secondThird);
                Console.SetCursorPosition(13, offset + i * hidYOffset);
                Console.Write($"= {inCount} {GetNeuronColor(hn)} {1} {synapseBrush}");
                i++;
            }

            Console.SetCursorPosition(29, offset + outYOffset);
            Console.Write($"= {hiddenCount} {GetNeuronColor(neuralNet.OutputLayer.Neurons[0])} 1 --" );

            return offset + height + 2;
        }

        private static string GetNeuronColor(HiddenNeuron neuron) =>
            neuron.Outputs[0].Signal > 0 ? ActiveNeuron : InactiveNeuron;

        private static string GetNeuronColor(OutputNeuron neuron) =>
            neuron.Outputs[0].Signal > 0 ? ActiveNeuron : InactiveNeuron;

        private static string GetSynapseBrush(double current, double firstThird, double secondThird)
        {
            if (current < firstThird)
            {
                return "··";
            } 
            else if (current > secondThird)
            {
                return "--";
            }
            else
            {
                return "==";
            }
        }
    }
}
