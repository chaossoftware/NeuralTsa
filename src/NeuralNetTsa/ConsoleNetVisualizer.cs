using NeuralNetTsa.Configuration;
using NeuralNetTsa.NeuralNet;
using NeuralNetTsa.NeuralNet.Entities;
using System;
using System.Linq;

namespace NeuralNetTsa;

public sealed class ConsoleNetVisualizer
{
    private const int InfoBorderOffset = 47;
    private const int RightPartOffset = 50;
    private const int ReportOffset = 20;
    private const int TopRightOffset = 3;

    private const string InactiveNeuron = "░░";
    private const string ActiveNeuron = "██";
    private const string NeutralNeuron = "▒▒";
    private const string NeutralSynapse = ">> ";
    private const int MinOffsetY = 1;

    private readonly string _delimiter = new string('_', InfoBorderOffset - 2);

    private readonly int _inCount;
    private readonly int _hiddenCount;
    private readonly int _maxItemsCount;
    private readonly int _inOffsetY;
    private readonly int _hiddenOffsetY;
    private readonly int _outOffsetY;

    private readonly ChaosNeuralNet _neuralNet;

    public ConsoleNetVisualizer(ChaosNeuralNet neuralNet)
    {
        _neuralNet = neuralNet;

        _inCount = neuralNet.Params.Dimensions;
        _hiddenCount = neuralNet.Params.Neurons;
        _maxItemsCount = Math.Max(_inCount, _hiddenCount);

        _inOffsetY = _maxItemsCount / _inCount + MinOffsetY;
        _hiddenOffsetY = _maxItemsCount / _hiddenCount + MinOffsetY;
        _outOffsetY = _maxItemsCount / neuralNet.OutputLayer.Neurons.Length + MinOffsetY;
    }

    public void ReportCycle(ChaosNeuralNet net)
    {
        double error = net.OutputLayer.Neurons[0].ShortMemory[0];
        string currentIteration = net.current.ToString().PadRight(10);

        Console.SetCursorPosition(RightPartOffset, 0);
        Console.WriteLine($"Epoch {net.successCount}: {currentIteration:#,#}");
        Console.SetCursorPosition(RightPartOffset, 2);
        Console.WriteLine($"ε = {net.Epsilon:e}");
        Console.SetCursorPosition(RightPartOffset, 3);
        Console.WriteLine($"e = {error:e}");
        Visualize();
        int offset = ReportOffset;
        Console.SetCursorPosition(0, offset);
    }

    public void PrintNetParams(NeuralNetParameters neuralNetParams)
    {
        Console.WriteLine(_delimiter);

        Console.WriteLine(neuralNetParams.GetInfoFull());

        for (int i = 3; i < ReportOffset - 2; i++)
        {
            Console.SetCursorPosition(InfoBorderOffset - 2, i);
            Console.Write("|");
        }

        Console.SetCursorPosition(0, ReportOffset - 3);
        Console.WriteLine(_delimiter);
    }

    private void Visualize()
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
            Console.SetCursorPosition(RightPartOffset, i * _inOffsetY + TopRightOffset);
            Console.Write($"[] {synapseBrush}");
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
            Console.SetCursorPosition(RightPartOffset + 7, i * _hiddenOffsetY + TopRightOffset);
            Console.Write($"{GetNeuronColor(hn)} {synapseBrush}");
            i++;
        }

        Console.SetCursorPosition(RightPartOffset + 14, _outOffsetY + TopRightOffset);
        Console.Write($"{GetNeuronColor(_neuralNet.OutputLayer.Neurons[0])} {NeutralSynapse}");
    }

    private static string GetNeuronColor(HiddenNeuron neuron) =>
        neuron.Outputs[0].Signal > 0 ? ActiveNeuron : InactiveNeuron;

    private static string GetNeuronColor(OutputNeuron neuron) =>
        neuron.Outputs[0].Signal > 0 ? ActiveNeuron : InactiveNeuron;

    private static string GetSynapseBrush(double current, double firstThird, double secondThird)
    {
        if (current < firstThird)
        {
            return ">  ";
        } 
        else if (current > secondThird)
        {
            return NeutralSynapse;
        }
        else
        {
            return ">>>";
        }
    }
}
