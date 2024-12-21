using System;
using System.Collections.Generic;
using System.Diagnostics;
using ChaosSoft.Core.DataUtils;
using MersenneTwister;
using NeuralNetTsa.Configuration;
using NeuralNetTsa.NeuralNet.Entities;
using ChaosSoft.NeuralNetwork.Networks;
using BaseEntities = ChaosSoft.NeuralNetwork.Entities;
using NeuralNetTsa.NeuralNet.CustomActivation;

namespace NeuralNetTsa.NeuralNet;

public sealed class ChaosNeuralNet : INeuralNet
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

    public ChaosNeuralNet(NeuralNetParams taskParams, double[] array) 
    {
        InputLayer = new BaseEntities.Layer<InputNeuron>(taskParams.Dimensions);
        HiddenLayer = new BaseEntities.Layer<HiddenNeuron>(taskParams.Neurons);
        OutputLayer = new BaseEntities.Layer<OutputNeuron>(1);

        Connections = new List<PruneSynapse>[] 
        { 
            new List<PruneSynapse>(), 
            new List<PruneSynapse>() 
        };

        Params = taskParams;
        AdditionalNeuron = Params.ActFunction is ComplexActivationFunction;
        Init(array);
    }

    public delegate void NeuralNetEvent(ChaosNeuralNet network);

    public event NeuralNetEvent CycleComplete;

    public event NeuralNetEvent EpochComplete;

    public NeuralNetParams Params { get; set; }

    public BiasNeuron NeuronBias { get; set; }

    public BiasNeuron NeuronConstant { get; set; }

    public BaseEntities.Layer<InputNeuron> InputLayer { get; }

    public BaseEntities.Layer<HiddenNeuron> HiddenLayer { get; }

    public BaseEntities.Layer<OutputNeuron> OutputLayer { get; }

    public List<PruneSynapse>[] Connections { get; }

    public double Epsilon => Math.Abs(ddw);

    Stopwatch timer = Stopwatch.StartNew();

    public void Process()
    {
        while (successCount < Params.Trainings)
        {
            if (OutputLayer.Neurons[0].LongMemory[0] == 0)
            {
                OutputLayer.Neurons[0].ShortMemory[0] = 1000;
            }
            else
            {
                OutputLayer.Neurons[0].ShortMemory[0] = 10 * OutputLayer.Neurons[0].LongMemory[0];
                ddw = Math.Min(Params.MaxPertrubation, Math.Sqrt(OutputLayer.Neurons[0].LongMemory[0]));
            }

            #region "update memory with best results"

            //update memory with best results
            foreach (InputNeuron neuron in InputLayer.Neurons)
            {
                neuron.LongMemoryToShort();
            }

            NeuronConstant.LongMemoryToShort();

            foreach (HiddenNeuron neuron in HiddenLayer.Neurons)
            {
                neuron.LongMemoryToShort();
            }

            NeuronBias.LongMemoryToShort();

            // same for Activation function neuron if needed
            if (AdditionalNeuron)
            {
                (Params.ActFunction as ComplexActivationFunction).Neuron.LongMemoryToShort();

                if ((Params.ActFunction as ComplexActivationFunction).Neuron.ShortMemory[0] == 0)
                {
                    (Params.ActFunction as ComplexActivationFunction).Neuron.ShortMemory[0] = 1;
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
                        double dj = 1.0 / Math.Pow(2, minD5DivD * j);

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

                #region calculate 'mean square' error in prediction of all points

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

                if (e1 < OutputLayer.Neurons[0].ShortMemory[0])
                {
                    #region memorize current weights and new error 
                    improved++;
                    OutputLayer.Neurons[0].ShortMemory[0] = e1;

                    foreach (InputNeuron neuron in InputLayer.Neurons)
                    {
                        neuron.WeightsToShortMemory();
                    }

                    foreach (HiddenNeuron neuron in HiddenLayer.Neurons)
                    {
                        neuron.WeightsToShortMemory();
                    }

                    NeuronBias.WeightsToShortMemory();
                    NeuronConstant.WeightsToShortMemory();

                    // same for Activation function neuron if needed
                    if (AdditionalNeuron)
                    {
                        (Params.ActFunction as ComplexActivationFunction).Neuron.WeightsToShortMemory();
                    }

                    #endregion
                }
                else if (ddw > 0 && improved == 0)
                {
                    ddw = -ddw;     //Try going in the opposite direction
                }
                //Reseed the random if the trial failed
                else
                {
                    seed = NeuronRandomizer.Randomizer.Next(int.MaxValue);
                    //seed = (int)(1 / Math.Sqrt(e1));

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

                NeuronRandomizer.Randomizer = Randoms.Create(seed, RandomType.FastestDouble);
    
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

                if (OutputLayer.Neurons[0].ShortMemory[0] > OutputLayer.Neurons[0].LongMemory[0] && OutputLayer.Neurons[0].LongMemory[0] != 0)
                {
                    continue;
                }

                // same for Activation function neuron if needed
                if (AdditionalNeuron)
                {
                    (Params.ActFunction as ComplexActivationFunction).Neuron.ShortMemoryToWeights();
                }

                #region Mark the weak connections for pruning

                if (Params.Pruning != 0)
                {
                    for (int i = 0; i < Params.Neurons; i++)
                    {
                        for (int j = 0; j < Params.Dimensions; j++)
                        {
                            double aBest = InputLayer.Neurons[j].ShortMemory[i];
                            double bBest = HiddenLayer.Neurons[i].ShortMemory[0];

                            if (aBest != 0 && Math.Abs(aBest * bBest) < tenPowNegativePruning)
                            {
                                InputLayer.Neurons[j].Outputs[i].Prune = true;
                            }
                        }

                        if (NeuronConstant.ShortMemory[i] != 0 && Math.Abs(NeuronConstant.ShortMemory[i] * HiddenLayer.Neurons[i].ShortMemory[0]) < tenPowNegativePruning)
                        {
                            NeuronConstant.Outputs[i].Prune = true;
                        }
                    }
                }

                #endregion
            }

            #region "Save best weights"

            //Save best weights
            foreach (InputNeuron neuron in InputLayer.Neurons)
            {
                neuron.ShortMemoryToLong();
            }

            foreach (HiddenNeuron neuron in HiddenLayer.Neurons)
            {
                neuron.ShortMemoryToLong();
            }

            NeuronBias.ShortMemoryToLong();
            NeuronConstant.ShortMemoryToLong();

            // same for Activation function neuron if needed
            if (AdditionalNeuron)
            {
                (Params.ActFunction as ComplexActivationFunction).Neuron.ShortMemoryToLong();
                (Params.ActFunction as ComplexActivationFunction).Neuron.LongMemoryToWeights();
            }

            #endregion

            successCount++;

            try
            {
                Console.SetCursorPosition(0, 18);
                Console.WriteLine($"Epoch duration: {timer.ElapsedMilliseconds:#,#} ms\n");
                EpochComplete.Invoke(this);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error during performing calculations:\n" + ex);
            }

            timer.Restart();
        }
    }

    /// <summary>
    /// Init neural network parameters
    /// all arrays should have +1 length
    /// </summary>
    private void Init(double[] sourceArray)
    {
        nmax = sourceArray.Length;
        double xmax = Vector.MaxAbs(sourceArray);

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

    public void ConstructNetwork()
    {
        NeuronRandomizer.Randomizer = Randoms.FastestDouble;

        // init input layer
        for (int i = 0; i < Params.Dimensions; i++)
        {
            InputNeuron neuron = new(Params.Nudge, Params.Neurons);
            neuron.Inputs.Add(new PruneSynapse(i, i, 1));
            InputLayer.Neurons[i] = neuron;
        }

        // init constant neuron
        NeuronConstant = new BiasNeuron(Params.Nudge, Params.Neurons);

        // init hidden layer
        HiddenNeuron.Function = Params.ActFunction;

        for (int i = 0; i < Params.Neurons; i++)
        {
            HiddenNeuron neuron = new(Params.Nudge, 1);
            HiddenLayer.Neurons[i] = neuron;
        }

        // init bias neuron
        NeuronBias = new BiasNeuron(Params.Nudge, 1);

        // init output layer
        OutputNeuron outNeuron = new(Params.Nudge, 1);
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
            PruneSynapse constantSynapse = new(Params.Dimensions, i);
            NeuronConstant.Outputs.Add(constantSynapse);
            HiddenLayer.Neurons[i].BiasInput = constantSynapse;
        }

        //Connect hidden and output layer neurons
        for (int i = 0; i < Params.Neurons; i++)
        {
            Connections[1].Add(new PruneSynapse(i, 0));
        }

        //Connect bias output and output neuron bias inputs
        PruneSynapse biasSynapse = new(Params.Neurons, 1);
        NeuronBias.Outputs.Add(biasSynapse);
        OutputLayer.Neurons[0].BiasInput = biasSynapse;


        foreach (var synapse in Connections[0])
        {
            InputLayer.Neurons[synapse.InIndex].Outputs.Add(synapse);
            HiddenLayer.Neurons[synapse.OutIndex].Inputs.Add(synapse);
        }

        foreach (var synapse in Connections[1])
        {
            HiddenLayer.Neurons[synapse.InIndex].Outputs.Add(synapse);
            OutputLayer.Neurons[synapse.OutIndex].Inputs.Add(synapse);
        }
    }
}
