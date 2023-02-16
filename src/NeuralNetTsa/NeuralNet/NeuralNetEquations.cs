using System;
using ChaosSoft.NumericalMethods.Equations;
using NeuralNetTsa.NeuralNet.Entities;
using ChaosSoft.NeuralNetwork.Activation;

namespace NeuralNetTsa.NeuralNet;

public class NeuralNetEquations : SystemBase
{
    private readonly IActivationFunction Activation_Function;
    private readonly int Neurons;

    public NeuralNetEquations(int dimensions, int neurons, IActivationFunction activationFunction) 
        : base(dimensions)
    {
        Rows += dimensions;
        Neurons = neurons;
        Activation_Function = activationFunction;
    }

    public override string Name => "Neural Net";

    public override void GetDerivatives(double[,] current, double[,] derivs) =>
        throw new NotImplementedException();

    public double[,] Derivs(double[,] x, double[,] a, double[] b, double bias, BiasNeuron constant) 
    {
        
        double[] df = new double[Count];
        double[,] xnew = new double[Rows, Count];
        double arg;

        /*
         * Nonlinear neural net equations:
         */
        xnew[0, 0] = bias;

        for (int i = 0; i < Neurons; i++) 
        {
            arg = constant.Outputs[i].Weight;

            for (int j = 0; j < Count; j++)
            {
                arg += a[i, j] * x[0, j];
            }

            xnew[0, 0] += b[i] * Activation_Function.Phi(arg);
        }

        for (int j = 1; j < Count; j++)
        {
            xnew[0, j] = x[0, j - 1];
        }

        /*
         * Linearized neural net equations:
         */
        for (int k = 0; k < Count; k++) 
        {
            df[k] = 0;

            for (int i = 0; i < Neurons; i++) 
            {
                arg = constant.Outputs[i].Weight;

                for (int j = 0; j < Count; j++)
                {
                    arg += a[i, j] * x[0, j];
                }

                df[k] += b[i] * a[i, k] * Activation_Function.Dphi(arg);
            }
        }

        for (int k = 0; k < Count; k++) 
        {
            xnew[1, k] = 0;

            for (int j = 0; j < Count; j++)
            {
                xnew[1, k] += df[j] * x[j + 1, k];//xnew(k) + df(j) * x(k + d * (j - 1))
            }
        }

        for (int k = 2; k < Count + 1; k++) 
        {
            for (int j = 0; j < Count; j++) 
            {
                xnew[k, j] = x[k - 1, j];
            }
        }

        return xnew;
    }

    public override void SetInitialConditions(double[,] current) =>
        throw new NotImplementedException();

    public void Init(double[,] x, double[] xdata) 
    {
        // initial conditions for nonlinear map
        for (int i = 0; i < Count; i++) 
        {
            x[0, i] = xdata[Count - i - 1];   //was xdata[dimensions - i + 1]
        }

        // initial conditions for linearized maps
        for (int i = 1; i < Rows; i++) 
        {
            x[i, i - 1] = 1;
        }
    }

    public override string ToFileName() =>
        throw new NotImplementedException();

    public override string ToString() =>
        throw new NotImplementedException();

    public override void SetParameters(params double[] parameters) =>
        throw new NotImplementedException();
}
