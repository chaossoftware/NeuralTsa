using ChaosSoft.Core;
using ChaosSoft.NumericalMethods.Lyapunov;
using ChaosSoft.NumericalMethods.QrDecomposition;
using NeuralNetTsa.NeuralNet;
using NeuralNetTsa.Utils;
using System;

namespace NeuralNetTsa.Routines;

internal sealed class LeSpecCalculator
{
    public double[] Result { get; set; }

    public double[,] LeInTime { get; set; }

    public void Calculate(ChaosNeuralNet net)
    {
        int n = net.Params.Dimensions;
        int iterations = net.xdata.Length - n - 1;
        LeInTime = new double[n, iterations];

        NeuralNetEquations neuralNetEquations = new(net);
        ModifiedGrammSchmidt ort = new(n);

        ModifiedDiscreteSolver solver = new(neuralNetEquations, net.xdata);
        solver.SetInitialConditions(0, GetInitialConditions(net.xdata, n));
        solver.SetLinearInitialConditions(GetLinearInitialConditions(n));

        LeSpecBenettin leSpec = new(solver, iterations, ort, 1);

        leSpec.Calculate();

        Console.WriteLine("LEs = {0}\t\t\t", NumFormat.Format(leSpec.Result, Constants.LeNumFormat, " "));
        Console.WriteLine("Dky = {0}", NumFormat.Format(StochasticProperties.KYDimension(leSpec.Result), Constants.LeNumFormat));
        Console.WriteLine("Eks = {0}", NumFormat.Format(StochasticProperties.KSEntropy(leSpec.Result), Constants.LeNumFormat));
        Console.WriteLine("PVC = {0}", NumFormat.Format(StochasticProperties.PhaseVolumeContractionSpeed(leSpec.Result), Constants.LeNumFormat));

        Result = leSpec.Result;
    }

    private static double[] GetInitialConditions(double[] xData, int n)
    {
        double[] conditions = new double[n];

        for (int i = 0; i < n; i++)
        {
            conditions[i] = xData[n - i - 1];   //was xdata[dimensions - i + 1]
        }

        return conditions;
    }

    private static double[,] GetLinearInitialConditions(int n)
    {
        double[,] conditions = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            conditions[i, i] = 1;
        }

        return conditions;
    }
}
