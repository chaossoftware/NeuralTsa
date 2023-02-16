using System;
using System.Globalization;
using System.Text;
using ChaosSoft.Core.IO;
using ChaosSoft.NumericalMethods.Lyapunov;
using NeuralNetTsa.NeuralNet;
using NeuralNetTsa.NeuralNet.Entities;

namespace NeuralNetTsa.Routines;

internal class DebugInfo
{
    public static void Write(ChaosNeuralNet net, double[] les, double lle)
    {
        double eBest = net.OutputLayer.Neurons[0].ShortMemory[0];
        
        StringBuilder debug = new StringBuilder()
            .AppendLine($"\nEpoch {net.successCount}")
            .AppendFormat(CultureInfo.InvariantCulture, "ε = {0:e}\n", net.Epsilon)
            .AppendFormat(CultureInfo.InvariantCulture, "e = {0:e}\n\n", eBest)
            .AppendLine($"LLE = {Format.General(lle)}")
            .AppendLine("LES = " + Format.General(les, " ", 6))
            .AppendLine($"Dky = {Format.General(StochasticProperties.KYDimension(les))}")
            .AppendLine($"Eks = {Format.General(StochasticProperties.KSEntropy(les))}")
            .AppendLine($"PVC = {Format.General(StochasticProperties.PhaseVolumeContractionSpeed(les))}")
            .AppendFormat(CultureInfo.InvariantCulture, "\nBias = {0:F6}\n\n", net.NeuronBias.ShortMemory[0]);

        for (int i = 0; i < net.Params.Neurons; i++)
        {
            debug.AppendFormat("Neuron {0}:\t", i + 1);
        }

        debug.AppendLine();

        Array.ForEach(net.NeuronConstant.ShortMemory, 
            memory => debug.Append(FormatNeuronValue(memory)));

        debug.AppendLine();

        foreach (InputNeuron neuron in net.InputLayer.Neurons)
        {
            Array.ForEach(neuron.ShortMemory,
                memory => debug.Append(FormatNeuronValue(memory)));

            debug.AppendLine();
        }

        debug.AppendLine();

        Array.ForEach(net.HiddenLayer.Neurons,
            neuron => debug.Append(FormatNeuronValue(neuron.ShortMemory[0])));

        debug.AppendLine("\n\n< < <");

        Logger.LogInfo(debug.ToString(), true);
    }

    private static string FormatNeuronValue(double value) =>
        string.Format(CultureInfo.InvariantCulture, "{0:F6}\t", value);
}
