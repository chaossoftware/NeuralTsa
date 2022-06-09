using NeuralNetTsa.NeuralNet;
using NeuralNetTsa.NeuralNet.Entities;
using System;
using System.Linq;

namespace NeuralNetTsa
{
    public class ConsoleNetVisualizer
    {
        private const string InactiveNeuron = "░░";
        private const string ActiveNeuron = "▓▓";
        private const string NeutralNeuron = "▒▒";
        private const string NeutralSynapse = "--";
        private const int MinOffsetY = 1;

        private readonly SciNeuralNet _neuralNet;

        private readonly int _inCount;
        private readonly int _hiddenCount;
        private readonly int _maxItemsCount;
        private readonly int _inOffsetY;
        private readonly int _hiddenOffsetY;
        private readonly int _outOffsetY;

        public ConsoleNetVisualizer(SciNeuralNet neuralNet)
        {
            _neuralNet = neuralNet;

            _inCount = neuralNet.Params.Dimensions;
            _hiddenCount = neuralNet.Params.Neurons;
            _maxItemsCount = Math.Max(_inCount, _hiddenCount);

            _inOffsetY = _maxItemsCount / _inCount + MinOffsetY;
            _hiddenOffsetY = _maxItemsCount / _hiddenCount + MinOffsetY;
            _outOffsetY = _maxItemsCount / neuralNet.OutputLayer.Neurons.Length + MinOffsetY;
        }

        public int Visualize(int offset)
        {
            double maxSynapseValue = _neuralNet.Connections[0].Max(s => s.Signal);
            double minSynapseValue = _neuralNet.Connections[0].Min(s => s.Signal);

            double segment = (maxSynapseValue - minSynapseValue) / 3d;
            double lowSynapseTreshold = minSynapseValue + segment;
            double highSynapseTreshold = minSynapseValue + 2d * segment;

            int i = 1;

            foreach (InputNeuron inn in _neuralNet.InputLayer.Neurons)
            {
                string synapseBrush = GetSynapseBrush(inn.Outputs[0].Signal, lowSynapseTreshold, highSynapseTreshold);
                Console.SetCursorPosition(0, offset + i * _inOffsetY);
                Console.Write($"■ {_hiddenCount} {synapseBrush}");
                i++;
            }

            maxSynapseValue = _neuralNet.Connections[1].Max(s => s.Signal);
            minSynapseValue = _neuralNet.Connections[1].Min(s => s.Signal);

            segment = (maxSynapseValue - minSynapseValue) / 3d;
            lowSynapseTreshold = minSynapseValue + segment;
            highSynapseTreshold = minSynapseValue + 2d * segment;
            i = 1;

            foreach (HiddenNeuron hn in _neuralNet.HiddenLayer.Neurons)
            {
                string synapseBrush = GetSynapseBrush(hn.Outputs[0].Signal, lowSynapseTreshold, highSynapseTreshold);
                Console.SetCursorPosition(13, offset + i * _hiddenOffsetY);
                Console.Write($"= {_inCount} {GetNeuronColor(hn)} 1 {synapseBrush}");
                i++;
            }

            Console.SetCursorPosition(29, offset + _outOffsetY);
            Console.Write($"= {_hiddenCount} {GetNeuronColor(_neuralNet.OutputLayer.Neurons[0])} 1 {NeutralSynapse}");

            return _maxItemsCount * 2 + offset + 2;
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
                return NeutralSynapse;
            }
            else
            {
                return "==";
            }
        }
    }
}
