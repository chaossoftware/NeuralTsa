using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuralNetTsa.NeuralNet.Entities;

namespace NeuralNetTsa.NeuralNet.Obsolete
{
    public struct Memory
    {
        public Memory(double shortMemory, double longMemory)
        {
            this.Short = shortMemory;
            this.Long = longMemory;
        }

        public double Short { get; set; }

        public double Long { get; set; }
    }

    public abstract class ObsoleteNeuron : ICloneable
    {
        public static Random Randomizer { get; set; }

        public PruneSynapse[] Inputs { get; set; }

        public PruneSynapse[] Outputs { get; set; }

        public PruneSynapse BiasInput { get; set; }

        public Memory[] Memory { get; set; }

        public double Nudge { get; set; }

        public virtual void CalculateWeight(int index, double perturbation)
        {
            Outputs[index].Weight = Memory[index].Short + perturbation * (Gauss2() - Nudge * Math.Sign(Memory[index].Short));
        }

        public virtual void CalculateWeight(int index, double perturbation, double lowerThan)
        {
            Outputs[index].Weight = Memory[index].Short;

            if (Randomizer.NextDouble() < lowerThan)
                Outputs[index].Weight += perturbation * (Gauss2() - Nudge * Math.Sign(Memory[index].Short));
        }

        public abstract void Process();

        public void BestToMemory()
        {
            for (int i = 0; i < Outputs.Length; i++)
            {
                Memory[i].Short = Memory[i].Long;
            }
        }

        public void WeightsToMemory()
        {
            for (int i = 0; i < Outputs.Length; i++)
            {
                Memory[i].Short = Outputs[i].Weight;
            }
        }

        public void MemoryToWeights()
        {
            for (int i = 0; i < Outputs.Length; i++)
            {
                Outputs[i].Weight = Memory[i].Short;
            }
        }

        public void BestToWeights()
        {
            for (int i = 0; i < Outputs.Length; i++)
            {
                Outputs[i].Weight = Memory[i].Long;
            }
        }

        public void MemoryToBest()
        {
            for (int i = 0; i < Outputs.Length; i++)
            {
                Memory[i].Long = Memory[i].Short;
            }
        }

        public abstract object Clone();

        /// <summary>
        /// Returns the product of two normally (Gaussian) distributed random 
        /// deviates with meanof zero and standard deviation of 1.0
        /// </summary>
        /// <returns></returns>
        private double Gauss2()
        {
            double v1, v2, arg;

            do
            {
                v1 = 2d * Randomizer.NextDouble() - 1d;
                v2 = 2d * Randomizer.NextDouble() - 1d;
                arg = v1 * v1 + v2 * v2;
            }
            while (arg >= 1d || arg == 0d);

            return v1 * v2 * (-2d + Math.Log(arg) / arg);
        }
    }
}
