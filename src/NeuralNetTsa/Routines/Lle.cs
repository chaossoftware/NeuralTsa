using ChaosSoft.Core;
using NeuralNetTsa.NeuralNet;
using System;

namespace NeuralNetTsa.Routines;

internal static class Lle
{
    private const double Perturbation = 1e-8; //Perturbation size

    /// <summary>
    /// Calculate the largest Lyapunov exponent
    /// </summary>
    /// <returns></returns>
    internal static double Calculate(ChaosNeuralNet net)
    {
        int dimensions = net.Params.Dimensions;
        int neurons = net.Params.Neurons;

        // precalculated
        double perturbationDivSqrtD = Perturbation / Math.Sqrt(dimensions);
        double perturbationSqr = Math.Pow(Perturbation, 2);

        long nmax = net.xdata.Length;

        double arg, x;
        double[] dx = new double[dimensions];
        double ltot = 0d;

        for (int j = 0; j < dimensions; j++)
        {
            dx[j] = perturbationDivSqrtD;
        }

        for (int k = dimensions; k < nmax; k++)
        {
            x = net.NeuronBias.Outputs[0].Weight;

            for (int i = 0; i < neurons; i++)
            {
                arg = net.NeuronConstant.Outputs[i].Weight;

                for (int j = 0; j < dimensions; j++)
                {
                    arg += net.InputLayer.Neurons[j].Outputs[i].Weight * net.xdata[k - j - 1];
                }

                x += net.HiddenLayer.Neurons[i].Outputs[0].Weight * net.Params.ActFunction.Phi(arg);
            }

            double xe = net.NeuronBias.Outputs[0].Weight;

            for (int i = 0; i < neurons; i++)
            {
                arg = net.NeuronConstant.Outputs[i].Weight;

                for (int j = 0; j < dimensions; j++)
                {
                    arg += net.InputLayer.Neurons[j].Outputs[i].Weight * (net.xdata[k - j - 1] + dx[j]);
                }

                xe += net.HiddenLayer.Neurons[i].Outputs[0].Weight * net.Params.ActFunction.Phi(arg);
            }

            double rs = 0;

            for (int j = dimensions - 2; j >= 0; j--)
            {
                rs += dx[j] * dx[j];
                dx[j + 1] = dx[j];
            }

            dx[0] = xe - x;
            rs += dx[0] * dx[0];
            rs = Math.Sqrt(rs / perturbationSqr);

            for (int j = 0; j < dimensions; j++)
            {
                dx[j] /= rs;
            }

            ltot += Math.Log(rs);
        }

        double lle = ltot / (nmax - dimensions);
        Console.WriteLine("LLE = " + NumFormat.Format(lle, Constants.LeNumFormat));

        return lle;
    }
}
