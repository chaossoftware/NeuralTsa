using MathLib.MathMethods.Lyapunov;
using NeuralNet.Entities;
using System;
using MathLib;
using DeepLearn.NeuralNetwork.Networks;

namespace NeuralNetwork
{
    public class NeuralNet : ThreeLayerNetwork<InputNeuron, HiddenNeuron, OutputNeuron>
    {
        public BenettinResult Task_Result;
        public NeuralNetParams Params;
        public NeuralNetEquations System_Equations;

        public InputNeuron[] NeuronsInput;
        public HiddenNeuron[] NeuronsHidden;
        public OutputNeuron NeuronOutput;
        public BiasNeuron NeuronBias;
        public BiasNeuron NeuronConstant;

        //----- input data
        private long nmax;  //lines in file
        public double[] xdata;

        //----- pre-calculated constants
        private double tenPowMinPruning;      
        private double minD5DivD;
        private double nmaxSubDXmaxPowE;

        private int neurons, dims;
        private int countnd = 0;    //Allows for introducing n and d gradually

        private double ddw;

        //counters
        public int _c, successCount;

        private int improved = 0;
        private int seed;

        private bool AdditionalNeuron;

        public NeuralNet(NeuralNetParams taskParams, double[] array) 
            : base(taskParams.Dimensions, taskParams.Neurons, 1)
        {
            Params = taskParams;
            AdditionalNeuron = Params.ActFunction.AdditionalNeuron;
            System_Equations = new NeuralNetEquations(Params.Dimensions, Params.Neurons, Params.ActFunction);
            Init(array);
        }


        public MyMethodDelegate LoggingMethod = null;
        public MyMethodDelegate EndCycleMethod = null;
        public delegate void MyMethodDelegate();

        public void InvokeMethodForNeuralNet(MyMethodDelegate method) {
            method.DynamicInvoke();
        }
        
        public void RunTask() {

            while (successCount < Params.Trainings) {

                if (NeuronOutput.Best[0] == 0) {
                    NeuronOutput.Memory[0] = Params.TestingInterval;
                }
                else {
                    NeuronOutput.Memory[0] = 10 * NeuronOutput.Best[0];
                    ddw = Math.Min(Params.MaxPertrubation, Math.Sqrt(NeuronOutput.Best[0]));
                }


                #region "update memory with best results"

                //update memory with best results
                foreach (InputNeuron neuron in NeuronsInput)
                    neuron.BestToMemory();
                NeuronConstant.BestToMemory();

                foreach (HiddenNeuron neuron in NeuronsHidden)
                    neuron.BestToMemory();
                NeuronBias.BestToMemory();

                // same for Activation function neuron if needed
                if (AdditionalNeuron)
                {
                    Params.ActFunction.Neuron.BestToMemory();
                    if (Params.ActFunction.Neuron.Memory[0] == 0)
                        Params.ActFunction.Neuron.Memory[0] = 1;
                }

                #endregion


                int N_MUL_D_MINCT_PLUS_1_PLUS_N_PLUS_1 = neurons * (dims - Params.ConstantTerm + 1) + neurons + 1;

                for (_c = 1; _c <= Params.CMax; _c++) {

                    if (Params.Pruning == 0)
                    {
                        foreach (InputNeuron neuron in NeuronsInput)
                            foreach (PruneSynapse synapse in neuron.Outputs)
                                synapse.Prune = false;

                        foreach (PruneSynapse synapse in NeuronConstant.Outputs)
                            synapse.Prune = false;
                    }
                        
                    
                    int prunes = 0;

                    for (int i = 0; i < neurons; i++)
                    {
                        for (int j = 0; j < dims; j++)
                            if (NeuronsInput[j].Outputs[i].Weight == 0)
                                prunes++;

                        if (NeuronConstant.Outputs[i].Weight == 0 && Params.ConstantTerm == 0)
                            prunes++;
                    }


                    //Probability of changing a given parameter at each trial
                    //1 / Sqrt(neurons * (dims - Task_Params.ConstantTerm + 1) + neurons + 1 - prunes)
                    double pc = 1d / Math.Sqrt(N_MUL_D_MINCT_PLUS_1_PLUS_N_PLUS_1 - prunes); 

                    if (Params.BiasTerm == 0)
                        NeuronBias.CalculateWeight(0, ddw);
                    else
                        NeuronBias.Outputs[0].Weight = 0;

                    for (int i = 0; i < neurons; i++) {

                        NeuronsHidden[i].CalculateWeight(0, ddw, 9 * pc);

                        if(Params.ConstantTerm == 0)
                        {
                            NeuronConstant.CalculateWeight(i, ddw, pc);

                            //This connection has been pruned
                            if (NeuronConstant.Outputs[i].Prune)
                                NeuronConstant.Outputs[i].Weight = 0;
                        }

                        for (int j = 0; j < dims; j++) {
                
                            //Reduce neighborhood for large j by a factor of 1-32
                            double dj = 1d / Math.Pow(2, minD5DivD * j);

                            NeuronsInput[j].CalculateWeight(i, ddw * dj, pc);
                
                            //This connection has been pruned
                            if (NeuronsInput[j].Outputs[i].Prune)
                                NeuronsInput[j].Outputs[i].Weight = 0; 
                        }
                    }


                    // same for Activation function neuron if needed
                    if (AdditionalNeuron) {
                        for (int j = 0; j < 7; j++) {
                            Params.ActFunction.Neuron.CalculateWeight(j, ddw, pc);
                        }
                    }

                    double e1 = 0;


                    #region "calculate 'mean square' error in prediction of all points"

                    for (int k = Params.Dimensions; k < nmax; k++) {
            
                        // get inputs from signal data
                        for (int j = 0; j < Params.Dimensions; j++)
                            NeuronsInput[j].Input = xdata[k - j - 1];

                        foreach (InputNeuron neuron in NeuronsInput)
                            neuron.ProcessInputs();

                        foreach (HiddenNeuron neuron in NeuronsHidden)
                            neuron.ProcessInputs();

                        NeuronOutput.ProcessInputs();

                        //Error in the prediction of the k-th data point
                        double ex = Math.Abs(NeuronOutput.Outputs[0].Signal - xdata[k]);
                        e1 += Math.Pow(ex, Params.ErrorsExponent);
                    }

                    //"Mean-square" error (even for e&<>2)
                    e1 = Math.Pow(e1 / nmaxSubDXmaxPowE, 2 / Params.ErrorsExponent);

                    #endregion


                    if (e1 < NeuronOutput.Memory[0]) {

                        improved ++;
                        NeuronOutput.Memory[0] = e1;

                        //memorize current weights
                        foreach (InputNeuron neuron in NeuronsInput)
                            neuron.WeightsToMemory();

                        foreach (HiddenNeuron neuron in NeuronsHidden)
                            neuron.WeightsToMemory();

                        NeuronBias.WeightsToMemory();
                        NeuronConstant.WeightsToMemory();


                        // same for Activation function neuron if needed
                        if (AdditionalNeuron)
                            Params.ActFunction.Neuron.WeightsToMemory();
                    }
                    else if (ddw > 0 && improved == 0) {
                        ddw = -ddw;     //Try going in the opposite direction
                    }
                    //Reseed the random if the trial failed
                    else {
                        seed = Neuron.Randomizer.Next(int.MaxValue);     //seed = (int) (1 / Math.Sqrt(e1));
            
                        if (improved > 0) {
                            ddw = Math.Min(Params.MaxPertrubation, (1 + improved / Params.TestingInterval) * Math.Abs(ddw));
                            improved = 0;
                        }
                        else
                            ddw = Params.Eta * Math.Abs(ddw);
                    }

                    Neuron.Randomizer = new Random(seed);
        
                    //Testing is costly - don't do it too often
                    if (_c % Params.TestingInterval != 0)
                        continue;


                    InvokeMethodForNeuralNet(LoggingMethod);
        

                    if(NeuronOutput.Memory[0] > NeuronOutput.Best[0] && NeuronOutput.Best[0] != 0)
                        continue;


                    // same for Activation function neuron if needed
                    if (AdditionalNeuron) {
                        Params.ActFunction.Neuron.MemoryToWeights();
                    }

     		
                    //Mark the weakconnections for pruning
                    if (Params.Pruning != 0) 
                        for (int i = 0; i < Params.Neurons; i++)
                        {
                            for (int j = 0; j < Params.Dimensions; j++)
                            {
                                double aBest = NeuronsInput[j].Memory[i];
                                double bBest = NeuronsHidden[i].Memory[0];
                                if (aBest != 0 && Math.Abs(aBest * bBest) < tenPowMinPruning)
                                    NeuronsInput[j].Outputs[i].Prune = true;
                            }

                            if (NeuronConstant.Memory[i] != 0 && Math.Abs(NeuronConstant.Memory[i] * NeuronsHidden[i].Memory[0]) < tenPowMinPruning)
                                NeuronConstant.Outputs[i].Prune = true;
                        }
                }


                if (++countnd % 2 != 0)
                    neurons = Math.Min(neurons + 1, Params.Neurons); //Increase the number of neurons slowly
                else
                    dims = Math.Min(dims + 1, Params.Dimensions); //And then increase the number of dimensions


                #region "Save best weights"

                //Save best weights
                foreach (InputNeuron neuron in NeuronsInput)
                    neuron.MemoryToBest();

                foreach (HiddenNeuron neuron in NeuronsHidden)
                    neuron.MemoryToBest();

                NeuronBias.MemoryToBest();
                NeuronConstant.MemoryToBest();

                // same for Activation function neuron if needed
                if (AdditionalNeuron) {
                    Params.ActFunction.Neuron.MemoryToBest();
                    Params.ActFunction.Neuron.BestToWeights();
                }

                #endregion


                successCount++;
                InvokeMethodForNeuralNet(EndCycleMethod);
            }
        }


        /// <summary>
        /// Init neural network parameters
        /// all arrays should have +1 length
        /// </summary>
        private void Init(double[] sourceArray) {

            nmax = sourceArray.Length;
            double xmax = Ext.countMaxAbs(sourceArray);

            // create array with data
            xdata = new double[nmax];
            Array.Copy(sourceArray, xdata, nmax);

            ConstructNetwork();

            tenPowMinPruning = Math.Pow(10, -Params.Pruning);
            minD5DivD = Math.Min(Params.Dimensions, 5) / Params.Dimensions;
            nmaxSubDXmaxPowE = (nmax - Params.Dimensions) * Math.Pow(xmax, Params.ErrorsExponent);
            
            neurons = Params.Neurons;
            dims = Params.Dimensions;

            ddw = Params.MaxPertrubation;
        }


        public override void ConstructNetwork()
        {
            //random = new Random();
            Neuron.Randomizer = new Random();

            // init input layer
            for (int i = 0; i < Params.Dimensions; i++)
            {
                var neuron = new InputNeuron(Params.Nudge);
                neuron.Memory = new double[Params.Neurons];
                neuron.Best = new double[Params.Neurons];
                neuron.Inputs.Add(new PruneSynapse(i, i, 1));
                this.InputLayer.Neurons[i] = neuron;
            }

            // init constant neuron
            NeuronConstant = new BiasNeuron(Params.Neurons, Params.Nudge);

            // init hidden layer
            HiddenNeuron.Function = Params.ActFunction;

            for (int i = 0; i < Params.Neurons; i++)
            {
                var neuron = new HiddenNeuron(Params.Nudge);
                neuron.Memory = new double[1];
                neuron.Best = new double[1];
                this.HiddenLayer.Neurons[i] = neuron;
            }

            // init bias neuron
            NeuronBias = new BiasNeuron(1, Params.Nudge);

            // init output layer
            var outNeuron = new OutputNeuron(Params.Nudge);
            outNeuron.Memory = new double[1];
            outNeuron.Best = new double[1];
            outNeuron.Outputs.Add(new PruneSynapse(0, 0, 1));
            this.HiddenLayer.Neurons[0] = outNeuron;

            //Connect input and hidden layer neurons
            for (int i = 0; i < Params.Dimensions; i++)
                for(int j = 0; j < Params.Neurons; j++)
                {
                    PruneSynapse synapse = new PruneSynapse();
                    NeuronsInput[i].Outputs[j] = synapse;
                    NeuronsHidden[j].Inputs[i] = synapse;
                }


            //Connect constant and hidden neurons bias inputs
            
            for (int i = 0; i < Params.Neurons; i++)
            {
                PruneSynapse constantSynapse = new PruneSynapse();
                NeuronConstant.Outputs[i] = constantSynapse;
                NeuronsHidden[i].BiasInput = constantSynapse;
            }
                

            //Connect hidden and output layer neurons
            for (int i = 0; i < Params.Neurons; i++)
            {
                PruneSynapse synapse = new PruneSynapse();
                NeuronsHidden[i].Outputs[0] = synapse;
                NeuronOutput.Inputs[i] = synapse;
            }

            //Connect constant and hidden neurons bias inputs
            PruneSynapse biasSynapse = new PruneSynapse();
            NeuronBias.Outputs[0] = biasSynapse;
            NeuronOutput.BiasInput = biasSynapse;
        }

        public override object Clone()
        {
            throw new NotImplementedException();
        }

        public override void Process()
        {
            throw new NotImplementedException();
        }
    }
}
