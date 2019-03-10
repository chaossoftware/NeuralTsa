using MathLib;
using MathLib.Data;
using MathLib.DrawEngine.Charts;
using MathLib.MathMethods.Lyapunov;
using MathLib.MathMethods.Orthogonalization;
using NeuralNet.Entities;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using MathLib.DrawEngine;

namespace NeuralNetwork
{
    internal class Calculations
    {
        private const double Perturbation = 1e-8; //Perturbation size

        private double perturbationDivSqrtD;
        private double perturbationSqr; //Pertrubation^2
        private NeuralNetParameters parameters;
        private BenettinResult benettin;
        private NeuralNetEquations systemEquations;

        public Visualizer Visualizator { get; set; }

        public Calculations(NeuralNetParameters parameters)
        {
            this.parameters = parameters;
            perturbationSqr = Math.Pow(Perturbation, 2);
            perturbationDivSqrtD = Perturbation / Math.Sqrt(parameters.Dimensions);
            systemEquations = new NeuralNetEquations(parameters.Dimensions, parameters.Neurons, parameters.ActFunction);
            this.Visualizator = new Visualizer(new Size(480, 848));
            this.Visualizator.NeuralAnimation = new Animation();
        }

        public void LogCycle(SciNeuralNet net)
        {
            this.Visualizator.NeuralAnimation.AddFrame(Visualizator.DrawBrain(net));
            Console.WriteLine("{0}\tE: {1:0.#####e-0}", net.current, net.OutputLayer.Neurons[0].Memory[0]);
        }

        public void PerformCalculations(SciNeuralNet net)
        {
            double lle = CalculateLargestLyapunovExponent(net);
            Console.WriteLine("\nLLE = {0:F5}\n\n", lle);

            benettin = CalculateLyapunovSpectrum(net.xdata, net.InputLayer.Neurons, net.HiddenLayer.Neurons, systemEquations, net.NeuronConstant, net.NeuronBias);

            ConstructAttractor(net.xdata, net.InputLayer.Neurons, net.HiddenLayer.Neurons, net.OutputLayer.Neurons[0], net.NeuronConstant, net.NeuronBias);
            Prediction(net.xdata, net.InputLayer.Neurons, net.HiddenLayer.Neurons, net.Params.PtsToPredict, net.NeuronConstant, net.NeuronBias);

            NeuralOutput.SaveDebugInfoToFile(net.OutputLayer.Neurons[0].Memory[0], benettin, lle, net.InputLayer.Neurons, net.OutputLayer.Neurons[0], net.HiddenLayer.Neurons, net.NeuronConstant, net.NeuronBias);

            Visualizator.DrawBrain(net).Save(NeuralOutput.NetworkPlotPlotFileName, ImageFormat.Png);
        }

        private BenettinResult CalculateLyapunovSpectrum(double[] xdata, InputNeuron[] inputs, HiddenNeuron[] hiddenNeurons, NeuralNetEquations systemEquations, BiasNeuron constant, BiasNeuron bias) {

            int Dim = systemEquations.EquationsCount;
            int DimPlusOne = systemEquations.TotalEquationsCount;

            Orthogonalization Ort = new ModifiedGrammSchmidt(Dim);
            BenettinMethod lyap = new BenettinMethod(Dim);
            BenettinResult result;

            double time = 0;                 //time
            int irate = 1;                   //integration steps per reorthonormalization
            int io = xdata.Length - Dim - 1;     //number of iterations of the Map

            double[,] x = new double[DimPlusOne, Dim];
            double[,] xnew = new double[DimPlusOne, Dim];
            double[,] v = new double[DimPlusOne, Dim];
            double[] znorm = new double[Dim];

            double[,] leInTime = new double[Dim, io];

            systemEquations.Init(v, xdata);

            for (int m = 0; m < io; m++) {
                for (int j = 0; j < irate; j++) {

                    Array.Copy(v, x, v.Length);

                    //Use actual data rather than iterated data
                    for (int i = 0; i < Dim; i++)
                        x[0, i] = xdata[Dim - i + m - 1];

                    xnew = systemEquations.Derivs(x, Get2DArray(hiddenNeurons, inputs, constant), Get1DArray(hiddenNeurons), bias.Outputs[0].Weight, constant);
                    Array.Copy(xnew, v, xnew.Length);

                    time++;
                }

                Ort.Perform(v, znorm);
                lyap.calculateLE(znorm, time);

                for (int k = 0; k < Dim; k++) {
                    if (znorm[k] > 0) {
                        leInTime[k, m] = Math.Log(znorm[k]);
                    }
                }
            }

            result = lyap.GetResults();
            result.LeSpectrumInTime = leInTime;
            NeuralOutput.CreateLeInTimeFile(leInTime);

            return result;
        }

        private double[,] Get2DArray(HiddenNeuron[] neurons, InputNeuron[] inputs, BiasNeuron constant)
        {
            double[,] arr = new double[neurons.Length, inputs.Length];

            for (int i = 0; i < neurons.Length; i++)
                for (int j = 0; j < inputs.Length; j++)
                    arr[i, j] = inputs[j].Outputs[i].Weight;

            return arr;
        }

        private double[] Get1DArray(HiddenNeuron[] neurons)
        {
            double[] arr = new double[neurons.Length];

            for (int i = 0; i < neurons.Length; i++)
                    arr[i] = neurons[i].Outputs[0].Weight;

            return arr;
        }

        /// <summary>
        /// Calculate the largest Lyapunov exponent
        /// </summary>
        /// <returns></returns>
        private double CalculateLargestLyapunovExponent(SciNeuralNet net)
        {
            long nmax = net.xdata.Length;

            double _arg, x;
            double[] dx = new double[parameters.Dimensions];

            for (int j = 0; j < parameters.Dimensions; j++)
                dx[j] = perturbationDivSqrtD;

            double _ltot = 0d;

            for (int k = parameters.Dimensions; k < nmax; k++)
            {
                x = net.NeuronBias.Outputs[0].Weight;
                for (int i = 0; i < parameters.Neurons; i++)
                {
                    _arg = net.NeuronConstant.Outputs[i].Weight;

                    for (int j = 0; j < parameters.Dimensions; j++)
                        _arg += net.InputLayer.Neurons[j].Outputs[i].Weight * net.xdata[k - j - 1];

                    x += net.HiddenLayer.Neurons[i].Outputs[0].Weight * parameters.ActFunction.Phi(_arg);
                }

                double xe = net.NeuronBias.Outputs[0].Weight;

                for (int i = 0; i < parameters.Neurons; i++)
                {
                    _arg = net.NeuronConstant.Outputs[i].Weight;

                    for (int j = 0; j < parameters.Dimensions; j++)
                        _arg += net.InputLayer.Neurons[j].Outputs[i].Weight * (net.xdata[k - j - 1] + dx[j]);

                    xe += net.HiddenLayer.Neurons[i].Outputs[0].Weight * parameters.ActFunction.Phi(_arg);
                }

                double rs = 0;

                for (int j = parameters.Dimensions - 2; j >= 0; j--)
                {
                    rs += dx[j] * dx[j];
                    dx[j + 1] = dx[j];
                }

                dx[0] = xe - x;
                rs += dx[0] * dx[0];
                rs = Math.Sqrt(rs / perturbationSqr);

                for (int j = 0; j < parameters.Dimensions; j++)
                    dx[j] /= rs;

                _ltot += Math.Log(rs);
            }

            return _ltot / (nmax - parameters.Dimensions);
        }

        private void ConstructAttractor(double[] xdata, InputNeuron[] inputs, HiddenNeuron[] hiddenNeurons, OutputNeuron outputNeuron, BiasNeuron constant, BiasNeuron bias) {

            long pts = 50000;

            double[] xt = new double[pts];
            double[] yt = new double[pts];
            double[] zt = new double[pts];
            double[] xlast = new double[parameters.Dimensions + 1];

            for (int j = 0; j <= parameters.Dimensions; j++)
                xlast[j] = xdata[xdata.Length - 1 - j];

            for (long t = 1; t < pts; t++)
            {
                double xnew = bias.Best[0];

                for (int i = 0; i < parameters.Neurons; i++)
                {
                    double _arg = constant.Best[i];

                    for (int j = 0; j < parameters.Dimensions; j++)
                        _arg += inputs[j].Best[i] * xlast[j - 1 + 1];

                    xnew += hiddenNeurons[i].Best[0] * parameters.ActFunction.Phi(_arg);
                }

                for (int j = parameters.Dimensions; j > 0; j--)
                    xlast[j] = xlast[j - 1];

                yt[t] = xnew;
                xlast[0] = xnew;
                xt[t] = xlast[1];

                zt[t] = parameters.Dimensions > 2 ? xlast[2] : 1d;
            }

            try
            {
                double[] constructedSignal = new double[xdata.Length];
                Array.Copy(xt, constructedSignal, xdata.Length);
                PlotObject signal = new SignalPlot(new Timeseries(constructedSignal), new Size(848, 480), 1);
                signal.Plot().Save(NeuralOutput.ReconstructedSignalPlotFileName, ImageFormat.Png);

                PlotObject poincare = new MapPlot(Ext.GeneratePseudoPoincareMapData(xt), new Size(848, 480), 1);
                poincare.Plot().Save(NeuralOutput.ReconstructedPoincarePlotFileName, ImageFormat.Png);
            }
            catch (Exception)
            {
            }

            NeuralOutput.Create3dModelFile(xt, yt, zt);

            NeuralOutput.CreateWavFile(yt);
        }

        private void Prediction(double[] xdata, InputNeuron[] input, HiddenNeuron[] hiddenNeurons, int pointsToPredict, BiasNeuron constant, BiasNeuron bias) {

            if (pointsToPredict == 0)
                return;

            double[] xpred = new double[pointsToPredict + parameters.Dimensions];
            double _xpred = 0;

            for (int j = 0; j < parameters.Dimensions; j++)
                xpred[parameters.Dimensions - j] = xdata[xdata.Length - 1 - j];

            for (int k = parameters.Dimensions; k < pointsToPredict + parameters.Dimensions; k++)
            {
                _xpred = bias.Best[0];
                for (int i = 0; i < parameters.Neurons; i++)
                {
                    double _arg = constant.Best[i];

                    for (int j = 0; j < parameters.Dimensions; j++)
                        _arg += input[j].Best[i] * xpred[k - j];

                    _xpred += hiddenNeurons[i].Best[0] * parameters.ActFunction.Phi(_arg);
                }

                xpred[k] = _xpred;
            }

            PlotObject prediction = new SignalPlot(new Timeseries(xpred), new Size(848, 480), 1);
            prediction.Plot().Save(NeuralOutput.PredictedSignalPlotFileName, ImageFormat.Png);

            NeuralOutput.CreatePredictedDataFile(xpred);
        }
    }
}
