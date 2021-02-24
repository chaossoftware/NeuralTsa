using System;
using NewMind.NeuralNet.Networks;
using MathLib;
using NeuralAnalyser.Configuration;
using NeuralAnalyser.NeuralNet.Activation;
using NeuralAnalyser.NeuralNet.Entities;

namespace NeuralAnalyser.NeuralNet
{
    public class SciNeuralNet : ThreeLayerNetwork<InputNeuron, HiddenNeuron, OutputNeuron, PruneSynapse>
    {
        //----- input data
        private long nmax;  //lines in file
        public double[] xdata;

        //----- pre-calculated constants
        private double tenPowNegativePruning;      
        private double minD5DivD;
        private double nmaxSubD_xmaxPowE;
        private int nMul_DSubCtAdd1_AddNAdd1;

        private double ddw;

        //counters
        public int current, successCount;

        private int improved = 0;
        private int seed;

        private bool AdditionalNeuron;

        public SciNeuralNet(NeuralNetParameters taskParams, double[] array) 
            : base(taskParams.Dimensions, taskParams.Neurons, 1)
        {
            Params = taskParams;
            AdditionalNeuron = Params.ActFunction is ComplexActivationFunction;
            Init(array);
        }

        public delegate void NeuralNetEvent(SciNeuralNet network);

        public event NeuralNetEvent CycleComplete;

        public event NeuralNetEvent EpochComplete;

        public NeuralNetParameters Params { get; set; }

        public BiasNeuron NeuronBias { get; set; }

        public BiasNeuron NeuronConstant { get; set; }

        public override void Process()
        {
            while (successCount < Params.Trainings)
            {
                if (OutputLayer.Neurons[0].Best[0] == 0)
                {
                    OutputLayer.Neurons[0].Memory[0] = Params.TestingInterval;
                }
                else
                {
                    OutputLayer.Neurons[0].Memory[0] = 10 * OutputLayer.Neurons[0].Best[0];
                    ddw = Math.Min(Params.MaxPertrubation, Math.Sqrt(OutputLayer.Neurons[0].Best[0]));
                }

                #region "update memory with best results"

                //update memory with best results
                foreach (InputNeuron neuron in InputLayer.Neurons)
                {
                    neuron.BestToMemory();
                }

                NeuronConstant.BestToMemory();

                foreach (HiddenNeuron neuron in HiddenLayer.Neurons)
                {
                    neuron.BestToMemory();
                }

                NeuronBias.BestToMemory();

                // same for Activation function neuron if needed
                if (AdditionalNeuron)
                {
                    (Params.ActFunction as ComplexActivationFunction).Neuron.BestToMemory();

                    if ((Params.ActFunction as ComplexActivationFunction).Neuron.Memory[0] == 0)
                    {
                        (Params.ActFunction as ComplexActivationFunction).Neuron.Memory[0] = 1;
                    }
                }

                #endregion

                for (current = 1; current <= Params.EpochInterval; current++)
                {
                    if (Params.Pruning == 0)
                    {
                        foreach (InputNeuron neuron in InputLayer.Neurons)
                        {
                            foreach (PruneSynapse synapse in neuron.Outputs)
                            {
                                synapse.Prune = false;
                            }
                        }

                        foreach (PruneSynapse synapse in NeuronConstant.Outputs)
                        {
                            synapse.Prune = false;
                        }
                    }
                    
                    int prunes = 0;

                    for (int i = 0; i < Params.Neurons; i++)
                    {
                        for (int j = 0; j < Params.Dimensions; j++)
                        {
                            if (InputLayer.Neurons[j].Outputs[i].Weight == 0)
                            {
                                prunes++;
                            }
                        }

                        if (NeuronConstant.Outputs[i].Weight == 0 && Params.ConstantTerm == 0)
                        {
                            prunes++;
                        }
                    }

                    //Probability of changing a given parameter at each trial
                    //1 / Sqrt(neurons * (dims - Task_Params.ConstantTerm + 1) + neurons + 1 - prunes)
                    double pc = 1d / Math.Sqrt(nMul_DSubCtAdd1_AddNAdd1 - prunes); 

                    if (Params.BiasTerm == 0)
                    {
                        NeuronBias.CalculateWeight(0, ddw);
                    }
                    else
                    {
                        NeuronBias.Outputs[0].Weight = 0;
                    }

                    for (int i = 0; i < Params.Neurons; i++)
                    {
                        HiddenLayer.Neurons[i].CalculateWeight(0, ddw, 9 * pc);

                        if (Params.ConstantTerm == 0)
                        {
                            NeuronConstant.CalculateWeight(i, ddw, pc);

                            //This connection has been pruned
                            NeuronConstant.Outputs[i].PruneIfMarked();
                        }

                        for (int j = 0; j < Params.Dimensions; j++)
                        {
                            //Reduce neighborhood for large j by a factor of 1-32
                            double dj = 1d / Math.Pow(2, minD5DivD * j);

                            InputLayer.Neurons[j].CalculateWeight(i, ddw * dj, pc);

                            //This connection has been pruned
                            InputLayer.Neurons[j].Outputs[i].PruneIfMarked();
                        }
                    }

                    // same for Activation function neuron if needed
                    if (AdditionalNeuron)
                    {
                        for (int j = 0; j < 7; j++)
                        {
                            (Params.ActFunction as ComplexActivationFunction).Neuron.CalculateWeight(j, ddw, pc);
                        }
                    }

                    double e1 = 0;

                    #region "calculate 'mean square' error in prediction of all points"

                    for (int k = Params.Dimensions; k < nmax; k++)
                    {
                        // get inputs from signal data
                        for (int j = 0; j < Params.Dimensions; j++)
                        {
                            InputLayer.Neurons[j].Inputs[0].Signal = xdata[k - j - 1];
                        }

                        InputLayer.Process();

                        HiddenLayer.Process();

                        OutputLayer.Process();

                        //Error in the prediction of the k-th data point
                        double ex = Math.Abs(OutputLayer.Neurons[0].Outputs[0].Signal - xdata[k]);
                        e1 += Math.Pow(ex, Params.ErrorsExponent);
                    }

                    //"Mean-square" error (even for e&<>2)
                    e1 = Math.Pow(e1 / nmaxSubD_xmaxPowE, 2d / Params.ErrorsExponent);

                    #endregion

                    if (e1 < OutputLayer.Neurons[0].Memory[0])
                    {
                        improved ++;
                        OutputLayer.Neurons[0].Memory[0] = e1;

                        //memorize current weights
                        foreach (InputNeuron neuron in InputLayer.Neurons)
                        {
                            neuron.WeightsToMemory();
                        }

                        foreach (HiddenNeuron neuron in HiddenLayer.Neurons)
                        {
                            neuron.WeightsToMemory();
                        }

                        NeuronBias.WeightsToMemory();
                        NeuronConstant.WeightsToMemory();

                        // same for Activation function neuron if needed
                        if (AdditionalNeuron)
                        {
                            (Params.ActFunction as ComplexActivationFunction).Neuron.WeightsToMemory();
                        }
                    }
                    else if (ddw > 0 && improved == 0)
                    {
                        ddw = -ddw;     //Try going in the opposite direction
                    }
                    //Reseed the random if the trial failed
                    else
                    {
                        seed = NeuronRandomizer.Randomizer.Next(int.MaxValue);     //seed = (int) (1 / Math.Sqrt(e1));

                        if (improved > 0)
                        {
                            ddw = Math.Min(Params.MaxPertrubation, (1 + improved / Params.TestingInterval) * Math.Abs(ddw));
                            improved = 0;
                        }
                        else
                        {
                            ddw = Params.Eta * Math.Abs(ddw);
                        }
                    }

                    NeuronRandomizer.Randomizer = new Random(seed);
        
                    //Testing is costly - don't do it too often
                    if (current % Params.TestingInterval != 0)
                    {
                        continue;
                    }

                    try
                    {
                        CycleComplete.Invoke(this);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine("Error during logging neural network cycle:\n" + ex);
                    }

                    if (OutputLayer.Neurons[0].Memory[0] > OutputLayer.Neurons[0].Best[0] && OutputLayer.Neurons[0].Best[0] != 0)
                    {
                        continue;
                    }

                    // same for Activation function neuron if needed
                    if (AdditionalNeuron)
                    {
                        (Params.ActFunction as ComplexActivationFunction).Neuron.MemoryToWeights();
                    }

                    //Mark the weak connections for pruning
                    if (Params.Pruning != 0)
                    {
                        for (int i = 0; i < Params.Neurons; i++)
                        {
                            for (int j = 0; j < Params.Dimensions; j++)
                            {
                                double aBest = InputLayer.Neurons[j].Memory[i];
                                double bBest = HiddenLayer.Neurons[i].Memory[0];

                                if (aBest != 0 && Math.Abs(aBest * bBest) < tenPowNegativePruning)
                                {
                                    InputLayer.Neurons[j].Outputs[i].Prune = true;
                                }
                            }

                            if (NeuronConstant.Memory[i] != 0 && Math.Abs(NeuronConstant.Memory[i] * HiddenLayer.Neurons[i].Memory[0]) < tenPowNegativePruning)
                            {
                                NeuronConstant.Outputs[i].Prune = true;
                            }
                        }
                    }
                }

                #region "Save best weights"

                //Save best weights
                foreach (InputNeuron neuron in InputLayer.Neurons)
                {
                    neuron.MemoryToBest();
                }

                foreach (HiddenNeuron neuron in HiddenLayer.Neurons)
                {
                    neuron.MemoryToBest();
                }

                NeuronBias.MemoryToBest();
                NeuronConstant.MemoryToBest();

                // same for Activation function neuron if needed
                if (AdditionalNeuron)
                {
                    (Params.ActFunction as ComplexActivationFunction).Neuron.MemoryToBest();
                    (Params.ActFunction as ComplexActivationFunction).Neuron.BestToWeights();
                }

                #endregion

                successCount++;

                try
                {
                    EpochComplete.Invoke(this);
                }
                catch (Exception ex)
                {
                    Console.WriteLine("Error during performing calculations:\n" + ex);
                }
            }
        }

        /// <summary>
        /// Init neural network parameters
        /// all arrays should have +1 length
        /// </summary>
        private void Init(double[] sourceArray)
        {
            nmax = sourceArray.Length;
            double xmax = Ext.CountMaxAbs(sourceArray);

            // create array with data
            xdata = new double[nmax];
            Array.Copy(sourceArray, xdata, nmax);

            ConstructNetwork();

            tenPowNegativePruning = Math.Pow(10, -Params.Pruning);
            minD5DivD = (double)Math.Min(Params.Dimensions, 5) / Params.Dimensions;
            nmaxSubD_xmaxPowE = (nmax - Params.Dimensions) * Math.Pow(xmax, Params.ErrorsExponent);
            nMul_DSubCtAdd1_AddNAdd1 = Params.Neurons * (Params.Dimensions - Params.ConstantTerm + 1) + Params.Neurons + 1;

            ddw = Params.MaxPertrubation;
        }

        public override void ConstructNetwork()
        {
            //random = new Random();
            NeuronRandomizer.Randomizer = new Random();

            // init input layer
            for (int i = 0; i < Params.Dimensions; i++)
            {
                var neuron = new InputNeuron(Params.Nudge);
                neuron.Memory = new double[Params.Neurons];
                neuron.Best = new double[Params.Neurons];
                neuron.Inputs.Add(new PruneSynapse(i, i, 1));
                InputLayer.Neurons[i] = neuron;
            }

            // init constant neuron
            NeuronConstant = new BiasNeuron(Params.Nudge);
            NeuronConstant.Memory = new double[Params.Neurons];
            NeuronConstant.Best = new double[Params.Neurons];

            // init hidden layer
            HiddenNeuron.Function = Params.ActFunction;

            for (int i = 0; i < Params.Neurons; i++)
            {
                var neuron = new HiddenNeuron(Params.Nudge);
                neuron.Memory = new double[1];
                neuron.Best = new double[1];
                HiddenLayer.Neurons[i] = neuron;
            }

            // init bias neuron
            NeuronBias = new BiasNeuron(Params.Nudge);
            NeuronBias.Memory = new double[1];
            NeuronBias.Best = new double[1];

            // init output layer
            var outNeuron = new OutputNeuron(Params.Nudge);
            outNeuron.Memory = new double[1];
            outNeuron.Best = new double[1];
            outNeuron.Outputs.Add(new PruneSynapse(0, 0, 1));
            OutputLayer.Neurons[0] = outNeuron;

            //Connect input and hidden layer neurons
            for (int i = 0; i < Params.Dimensions; i++)
            {
                for (int j = 0; j < Params.Neurons; j++)
                {
                    Connections[0].Add(new PruneSynapse(i, j));
                }
            }

            //Connect constant and hidden neurons bias inputs
            for (int i = 0; i < Params.Neurons; i++)
            {
                var constantSynapse = new PruneSynapse(Params.Dimensions, i);
                NeuronConstant.Outputs.Add(constantSynapse);
                HiddenLayer.Neurons[i].BiasInput = constantSynapse;
            }

            //Connect hidden and output layer neurons
            for (int i = 0; i < Params.Neurons; i++)
            {
                Connections[1].Add(new PruneSynapse(i, 0));
            }

            //Connect bias output and output neuron bias inputs
            PruneSynapse biasSynapse = new PruneSynapse(Params.Neurons, 1);
            NeuronBias.Outputs.Add(biasSynapse);
            OutputLayer.Neurons[0].BiasInput = biasSynapse;


            foreach (var synapse in Connections[0])
            {
                InputLayer.Neurons[synapse.IndexSource].Outputs.Add(synapse);
                HiddenLayer.Neurons[synapse.IndexDestination].Inputs.Add(synapse);
            }

            foreach (var synapse in Connections[1])
            {
                HiddenLayer.Neurons[synapse.IndexSource].Outputs.Add(synapse);
                OutputLayer.Neurons[synapse.IndexDestination].Inputs.Add(synapse);
            }
        }

        public override object Clone()
        {
            throw new NotImplementedException();
        }
    }
}
