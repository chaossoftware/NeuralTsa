using NeuralNetTsa.NeuralNet;
using NeuralNetTsa.NeuralNet.Entities;
using System.Linq;

namespace NeuralNetTsa.Utils;

internal static class NeuralNetDataConverter
{
    internal static double[,] GetL1Connections(ChaosNeuralNet neuralNet)
    {
        HiddenNeuron[] neurons = neuralNet.HiddenLayer.Neurons;
        InputNeuron[] inputs = neuralNet.InputLayer.Neurons;
        double[,] arr = new double[neurons.Length, inputs.Length];

        for (int i = 0; i < neurons.Length; i++)
        {
            for (int j = 0; j < inputs.Length; j++)
            {
                arr[i, j] = inputs[j].Outputs[i].Weight;
            }
        }

        return arr;
    }

    internal static double[] GetL2Connections(ChaosNeuralNet neuralNet) =>
        neuralNet.HiddenLayer.Neurons
        .Select(n => n.Outputs[0].Weight)
        .ToArray();
}
