﻿using System.Text;
using System.Globalization;

namespace NeuralNetwork
{
    /// <summary>
    /// Class Describing parameters of Neural network
    /// </summary>
    public class NeuralNetParams
    {
        public readonly double Eta = 0.999;                 //Learning rate
        public readonly long CMax = 1000000;                //Number of iterations
        public readonly int BiasTerm = 1;                   //0 for bias term; otherwise 1
        public readonly int ConstantTerm = 0;               //0 for constant term; otherwise 1
        public readonly double MaxPertrubation = 2;         //Maximum perturbation
        public readonly double Nudge = 0.5;                 //Amount to nudge the parameters back toward zero
        public readonly int Pruning = 0;                    //Pruning level (0 = no pruning)
        public readonly double TestingInterval = 1e4;       //Interval for testing neural net results
        
        public readonly int Neurons;                        //Neurons count
        public readonly int Dimensions;                     //System dimensions
        public readonly int ErrorsExponent;                 //Exponent of errors
        public readonly int Trainings;                      //Number of successful trainings to complete calculation
        public readonly int PtsToPredict;                   //Number of points to predict
        public readonly ActivationFunction ActFunction;     //Activation function

        public NeuralNetParams(int neurons, int dimensions, int errorsExponent, int trainings, 
            int ptsToPredict, ActivationFunction actFunction) {
            Neurons = neurons;
            Dimensions = dimensions;
            ErrorsExponent = errorsExponent;
            Trainings = trainings;
            PtsToPredict = ptsToPredict;
            ActFunction = actFunction;
        }

        public NeuralNetParams(int neurons, int dimensions, int errorsExponent, int trainings, 
            int ptsToPredict, ActivationFunction actFunction, double eta, long cmax, int biasTerm, 
            int constantTerm, double maxPertrubation, double nudge, int pruning, double testingInterval) {

            Neurons = neurons;
            Dimensions = dimensions;
            ErrorsExponent = errorsExponent;
            Trainings = trainings;
            PtsToPredict = ptsToPredict;
            ActFunction = actFunction;

            Eta = eta;
            CMax = cmax;
            BiasTerm = biasTerm;
            ConstantTerm = constantTerm;
            MaxPertrubation = maxPertrubation;
            Nudge = nudge;
            Pruning = pruning;
            TestingInterval = testingInterval;
        }

        public string GetInfoShort() => 
            new StringBuilder()
            .AppendFormat("Neurons: {0}\n", Neurons)
            .AppendFormat("Dimensions: {0}\n", Dimensions)
            .AppendFormat("Exponent of error: {0}\n", ErrorsExponent)
            .AppendFormat("Successful trainings count: {0}\n", Trainings)
            .AppendFormat("Point to predict: {0}\n", PtsToPredict)
            .AppendFormat("Activation Function: {0}\n", ActFunction.Name)
            .ToString();

        public string GetInfoFull() =>
            new StringBuilder()
            .Append(GetInfoShort() + "\n")
            .AppendFormat(CultureInfo.InvariantCulture, "Learning rate: {0}\n", Eta)
            .AppendFormat(CultureInfo.InvariantCulture, "Number of iterations: {0:0.#####e-0}\n", CMax)
            .AppendFormat("Bias termt: {0}\n", BiasTerm)
            .AppendFormat("Constant term: {0}\n", ConstantTerm)
            .AppendFormat(CultureInfo.InvariantCulture, "Maximum perturbation: {0}\n", MaxPertrubation)
            .AppendFormat(CultureInfo.InvariantCulture, "Amount to nudge the parameters back toward zero: {0}\n", Nudge)
            .AppendFormat("Pruning level: {0}\n", Pruning)
            .AppendFormat(CultureInfo.InvariantCulture, "Testing interval: {0:0.#####e-0}\n", TestingInterval)
            .ToString();
    }
}
