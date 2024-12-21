using ChaosSoft.NumericalMethods.Ode.Linearized;
using NeuralNetTsa.NeuralNet.Entities;
using ChaosSoft.NeuralNetwork.Activation;
using NeuralNetTsa.Utils;

namespace NeuralNetTsa.NeuralNet;

public sealed class NeuralNetEquations : ILinearizedOdeSys
{
    private readonly IActivationFunction ActivationFunction;
    private readonly int Neurons;

    private readonly double[,] _a;
    private readonly double[] _b;
    private readonly double _bias;
    private readonly BiasNeuron _constant;

    public NeuralNetEquations(ChaosNeuralNet neuralNet)
    {
        EqCount = neuralNet.Params.Dimensions;
        Neurons = neuralNet.Params.Neurons;
        ActivationFunction = neuralNet.Params.ActFunction;

        _a = NeuralNetDataConverter.GetL1Connections(neuralNet);
        _b = NeuralNetDataConverter.GetL2Connections(neuralNet);
        _bias = neuralNet.NeuronBias.Outputs[0].Weight;
        _constant = neuralNet.NeuronConstant;
    }

    public int EqCount { get; }

    // Nonlinear
    public void F(double t, double[] solution, double[] derivs)
    {
        double arg;
        derivs[0] = _bias;

        for (int i = 0; i < Neurons; i++)
        {
            arg = _constant.Outputs[i].Weight;

            for (int j = 0; j < EqCount; j++)
            {
                arg += _a[i, j] * solution[j];
            }

            derivs[0] += _b[i] * ActivationFunction.Phi(arg);
        }

        for (int j = 1; j < EqCount; j++)
        {
            derivs[j] = solution[j - 1];
        }
    }

    // Linearized
    public void F(double t, double[] solution, double[,] linearization, double[,] derivs)
    {
        double[] df = new double[EqCount];
        double arg;

        for (int k = 0; k < EqCount; k++)
        {
            df[k] = 0;

            for (int i = 0; i < Neurons; i++)
            {
                arg = _constant.Outputs[i].Weight;

                for (int j = 0; j < EqCount; j++)
                {
                    arg += _a[i, j] * solution[j];
                }

                df[k] += _b[i] * _a[i, k] * ActivationFunction.Dphi(arg);
            }
        }

        for (int k = 0; k < EqCount; k++)
        {
            derivs[0, k] = 0;

            for (int j = 0; j < EqCount; j++)
            {
                derivs[0, k] += df[j] * linearization[j, k];//xnew(k) + df(j) * x(k + d * (j - 1))
            }
        }

        for (int k = 1; k < EqCount; k++)
        {
            for (int j = 0; j < EqCount; j++)
            {
                derivs[k, j] = linearization[k - 1, j]; // xnew[k, j] = x[k - 1, j];
            }
        }
    }
}
