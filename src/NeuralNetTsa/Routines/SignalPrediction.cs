using System;
using System.Drawing;
using System.Linq;
using System.Text;
using ChaosSoft.Core;
using ChaosSoft.Core.IO;
using NeuralNetTsa.Configuration;
using NeuralNetTsa.NeuralNet;

namespace NeuralNetTsa.Routines;

internal static class SignalPrediction
{
    internal static void Make(ChaosNeuralNet net, OutputParams output, OutputPaths paths, double[] originalData)
    {
        int dimensions = net.Params.Dimensions;
        int neurons = net.Params.Neurons;
        int ptsToPredict = output.PtsToPredict;

        double[] xpred = new double[ptsToPredict + dimensions];
        double predPt;

        for (int j = 0; j < dimensions; j++)
        {
            xpred[dimensions - j - 1] = net.xdata[net.xdata.Length - 1 - j];
        }

        for (int k = dimensions; k < ptsToPredict + dimensions; k++)
        {
            predPt = net.NeuronBias.LongMemory[0];

            for (int i = 0; i < neurons; i++)
            {
                double arg = net.NeuronConstant.LongMemory[i];

                for (int j = 0; j < dimensions; j++)
                {
                    arg += net.InputLayer.Neurons[j].LongMemory[i] * xpred[k - j - 1];
                }

                predPt += net.HiddenLayer.Neurons[i].LongMemory[0] * net.Params.ActFunction.Phi(arg);
            }

            xpred[k] = predPt;
        }

        double[] xPredicted = xpred.Skip(dimensions).ToArray();

        StringBuilder prediction = new StringBuilder();
        Array.ForEach(xPredicted, x => prediction.AppendLine(NumFormat.Format(x)));
        FileUtils.CreateDataFile(paths.PredictFile, prediction.ToString());

        ScottPlot.Plot predictionPlot = new ScottPlot.Plot(output.PlotsSize.Width, output.PlotsSize.Height);
        predictionPlot.AddSignal(xPredicted);

        for (int i = 0; i < ptsToPredict; i++)
        {
            int index = originalData.Length - ptsToPredict + i;
            predictionPlot.AddMarker(i, originalData[index], size: 5, color: Color.IndianRed);
        }

        predictionPlot.XLabel("t");
        predictionPlot.YLabel("f(t)");
        predictionPlot.Title("Prediction");
        predictionPlot.SaveFig(paths.PredictedSignalPlotFile);
    }
}
