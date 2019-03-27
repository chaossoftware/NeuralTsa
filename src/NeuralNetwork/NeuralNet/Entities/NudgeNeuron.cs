using System;
using System.Collections.Generic;
using NewMind.NeuralNet.Base;

namespace NeuralAnalyser.NeuralNet.Entities
{
    public abstract class NudgeNeuron<N> : INeuron<N> where N : NudgeNeuron<N>
    {
        protected NudgeNeuron()
        {
            this.Inputs = new List<PruneSynapse>();
            this.Outputs = new List<PruneSynapse>();
        }

        public List<PruneSynapse> Inputs { get; set; }

        public List<PruneSynapse> Outputs { get; set; }

        public double[] Memory { get; set; }

        public double[] Best { get; set; }

        public double Nudge { get; set; }

        public virtual void CalculateWeight(int index, double pertrubation)
        {
            Outputs[index].Weight = Memory[index] + pertrubation * (Gauss2() - Nudge * Math.Sign(Memory[index]));
        }

        public virtual void CalculateWeight(int index, double pertrubation, double lowerThan)
        {
            Outputs[index].Weight = Memory[index];

            if(NeuronRandomizer.Randomizer.NextDouble() < lowerThan)
                Outputs[index].Weight += pertrubation * (Gauss2() - Nudge * Math.Sign(Memory[index]));
        }

        public void BestToMemory()
        {
            for(int i = 0; i < Outputs.Count; i++)
                Memory[i] = Best[i];
        }

        public void WeightsToMemory()
        {
            for (int i = 0; i < Outputs.Count; i++)
                Memory[i] = Outputs[i].Weight;
        }

        public void MemoryToWeights()
        {
            for (int i = 0; i < Outputs.Count; i++)
                Outputs[i].Weight = Memory[i];
        }

        public void BestToWeights()
        {
            for (int i = 0; i < Outputs.Count; i++)
                Outputs[i].Weight = Best[i];
        }

        public void MemoryToBest()
        {
            for (int i = 0; i < Outputs.Count; i++)
                Best[i] = Memory[i];
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

        public virtual void Process()
        {
            throw new NotImplementedException();
        }

        public virtual object Clone()
        {
            throw new NotImplementedException();
        }
    }
}
