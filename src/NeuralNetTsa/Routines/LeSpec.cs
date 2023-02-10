using ChaosSoft.Core.NumericalMethods.Lyapunov;
using ChaosSoft.Core.NumericalMethods.Orthogonalization;
using NeuralNetTsa.NeuralNet;
using NeuralNetTsa.NeuralNet.Entities;
using System;

namespace NeuralNetTsa.Routines;

internal static class LeSpec
{
    internal static double[] Calculate(ChaosNeuralNet net, NeuralNetEquations systemEquations)
    {
        int dim = systemEquations.Count;
        int dimPlusOne = systemEquations.Rows;

        ModifiedGrammSchmidt ort = new ModifiedGrammSchmidt(dim);
        BenettinMethod lyap = new BenettinMethod(dim);

        double time = 0;                 //time
        int irate = 1;                   //integration steps per reorthonormalization
        int io = net.xdata.Length - dim - 1;     //number of iterations of the Map

        double[,] x = new double[dimPlusOne, dim];
        double[,] xnew;// = new double[DimPlusOne, Dim];
        double[,] v = new double[dimPlusOne, dim];
        double[] znorm = new double[dim];

        double[,] leInTime = new double[dim, io];

        systemEquations.Init(v, net.xdata);

        for (int m = 0; m < io; m++)
        {
            for (int j = 0; j < irate; j++)
            {
                Array.Copy(v, x, v.Length);

                //Use actual data rather than iterated data
                for (int i = 0; i < dim; i++)
                {
                    x[0, i] = net.xdata[dim - i + m - 1];
                }

                xnew = systemEquations.Derivs(x, Get2DArray(net.HiddenLayer.Neurons, net.InputLayer.Neurons), Get1DArray(net.HiddenLayer.Neurons), net.NeuronBias.Outputs[0].Weight, net.NeuronConstant);
                Array.Copy(xnew, v, xnew.Length);

                time++;
            }

            ort.Perform(v, znorm);
            lyap.CalculateLyapunovSpectrum(znorm, time);

            for (int k = 0; k < dim; k++)
            {
                if (znorm[k] > 0)
                {
                    leInTime[k, m] = Math.Log(znorm[k]);
                }
            }
        }

        //TODO!!!
        //lyap.Result.SpectrumInTime = leInTime;

        return lyap.Result;
    }

    private static double[,] Get2DArray(HiddenNeuron[] neurons, InputNeuron[] inputs)
    {
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

    private static double[] Get1DArray(HiddenNeuron[] neurons)
    {
        double[] arr = new double[neurons.Length];

        for (int i = 0; i < neurons.Length; i++)
        {
            arr[i] = neurons[i].Outputs[0].Weight;
        }

        return arr;
    }
}
