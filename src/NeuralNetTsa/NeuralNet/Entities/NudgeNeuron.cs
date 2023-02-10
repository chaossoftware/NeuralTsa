using System;
using System.Collections.Generic;
using SciML.NeuralNetwork.Entities;

namespace NeuralNetTsa.NeuralNet.Entities;

public abstract class NudgeNeuron<N> : INeuron<N> where N : NudgeNeuron<N>
{
    protected NudgeNeuron(int capacity) : this(0d, capacity)
    {
    }

    protected NudgeNeuron(double nudge, int capacity)
    {
        Inputs = new List<PruneSynapse>();
        Outputs = new List<PruneSynapse>();
        ShortMemory = new double[capacity];
        LongMemory = new double[capacity];
        Nudge = nudge;
    }

    public List<PruneSynapse> Inputs { get; }

    public List<PruneSynapse> Outputs { get; }

    public double[] ShortMemory { get; }

    public double[] LongMemory { get; }

    public double Nudge { get; }

    public virtual void CalculateWeight(int index, double pertrubation)
    {
        Outputs[index].Weight = ShortMemory[index] + pertrubation * (Gauss2() - Nudge * Math.Sign(ShortMemory[index]));
    }

    public virtual void CalculateWeight(int index, double pertrubation, double lowerThan)
    {
        Outputs[index].Weight = ShortMemory[index];

        if(NeuronRandomizer.Randomizer.NextDouble() < lowerThan)
        {
            Outputs[index].Weight += pertrubation * (Gauss2() - Nudge * Math.Sign(ShortMemory[index]));
        }
    }

    public void LongMemoryToShort()
    {
        for(int i = 0; i < Outputs.Count; i++)
        {
            ShortMemory[i] = LongMemory[i];
        }
    }

    public void WeightsToShortMemory()
    {
        for (int i = 0; i < Outputs.Count; i++)
        {
            ShortMemory[i] = Outputs[i].Weight;
        }
    }

    public void ShortMemoryToWeights()
    {
        for (int i = 0; i < Outputs.Count; i++)
        {
            Outputs[i].Weight = ShortMemory[i];
        }
    }

    public void LongMemoryToWeights()
    {
        for (int i = 0; i < Outputs.Count; i++)
        {
            Outputs[i].Weight = LongMemory[i];
        }
    }

    public void ShortMemoryToLong()
    {
        for (int i = 0; i < Outputs.Count; i++)
        {
            LongMemory[i] = ShortMemory[i];
        }
    }


    /// <summary>
    /// Returns the product of two normally (Gaussian) distributed random 
    /// deviates with meanof zero and standard deviation of 1.0
    /// </summary>
    /// <returns></returns>
    private double Gauss2()
    {
        double v1, v2, _arg;
        do
        {
            v1 = 2d * NeuronRandomizer.Randomizer.NextDouble() - 1d;
            v2 = 2d * NeuronRandomizer.Randomizer.NextDouble() - 1d;
            _arg = v1 * v1 + v2 * v2;
        }
        while (_arg >= 1d || _arg == 0d);

        return v1 * v2 * (-2d + Math.Log(_arg) / _arg);
    }

    public abstract void Process();

    public virtual object Clone() =>
        throw new NotImplementedException();
}
