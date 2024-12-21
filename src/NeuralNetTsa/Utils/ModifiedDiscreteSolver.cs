using ChaosSoft.NumericalMethods.Ode.Linearized;
using System;

namespace NeuralNetTsa.Utils;

internal class ModifiedDiscreteSolver : LinearizedOdeSolverBase
{
    private readonly int _n;
    private readonly double[] _xData;
    private readonly double[] _derivs;
    private readonly double[,] _derivsLinear;

    /// <summary>
    /// Initializes a new instance of the <see cref="ModifiedDiscreteSolver"/> class for specified equations and time step.
    /// </summary>
    /// <param name="equations">equations system to solve</param>
    public ModifiedDiscreteSolver(ILinearizedOdeSys equations, double[] xData) : base(equations, 1)
    {
        _n = OdeSys.EqCount;
        _xData = xData;
        _derivs = new double[OdeSys.EqCount];
        _derivsLinear = new double[OdeSys.EqCount, OdeSys.EqCount];
    }

    /// <summary>
    /// Solves next step of system of equations.
    /// </summary>
    public override void NextStep()
    {
        // solver is discrete, so T is equal to currentIteration
        int currentIteration = (int)T;

        //Use actual data rather than iterated data
        for (int i = 0; i < _n; i++)
        {
            Solution[i] = _xData[_n - i + currentIteration - 1];
        }

        OdeSys.F(T, Solution, _derivs);
        LinearizedOdeSys.F(T, Solution, Linearization, _derivsLinear);

        Array.Copy(_derivs, Solution, Solution.Length);
        Array.Copy(_derivsLinear, Linearization, Linearization.Length);

        TimeIncrement();
    }
}
