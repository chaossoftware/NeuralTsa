using MathLib;
using MathLib.Data;
using MathLib.DrawEngine;
using MathLib.DrawEngine.Charts;
using MathLib.MathMethods.Lyapunov;
using MathLib.MathMethods.Orthogonalization;
using MathLib.NeuralNet.Entities;
using MathLib.NeuralNetwork;
using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace NeuralNetwork {
    class Calculations {

        private static double PERTRUB_SQR_D;
        private static double PERTRUB;                  //Perturbation size
        private static double PERTRUB_2;                //Pertrubation^2
        private static NeuralNetParams Task_Params;


        private static BenettinResult CalculateLyapunovSpectrum(double[] xdata, InputNeuron[] inputs, HiddenNeuron[] hiddenNeurons, NeuralNetEquations systemEquations, BiasNeuron constant, BiasNeuron bias) {

            int Dim = systemEquations.N;
            int DimPlusOne = systemEquations.NN;

            Orthogonalization Ort = new MGS(Dim);
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

                Ort.makeOrthogonalization(v, znorm);
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

        private static double[,] Get2DArray(HiddenNeuron[] neurons, InputNeuron[] inputs, BiasNeuron constant)
        {
            double[,] arr = new double[neurons.Length, inputs.Length];

            for (int i = 0; i < neurons.Length; i++)
                for (int j = 0; j < inputs.Length; j++)
                    arr[i, j] = inputs[j].Outputs[i].Weight;

            return arr;
        }

        private static double[] Get1DArray(HiddenNeuron[] neurons)
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
        private static double CalculateLargestLyapunovExponent(double[] xdata, InputNeuron[] inputs, HiddenNeuron[] hiddenNeurons, OutputNeuron outputNeuron, BiasNeuron constant, BiasNeuron bias) {
            long nmax = xdata.Length;

            double _arg, x;
            double[] dx = new double[Task_Params.Dimensions];

            for (int j = 0; j < Task_Params.Dimensions; j++)
                dx[j] = PERTRUB_SQR_D;

            double _ltot = 0d;

            for (int k = Task_Params.Dimensions; k < nmax; k++) {
                x = bias.Outputs[0].Weight;
                for (int i = 0; i < Task_Params.Neurons; i++) {
                    _arg = constant.Outputs[i].Weight;

                    for (int j = 0; j < Task_Params.Dimensions; j++)
                        _arg += inputs[j].Outputs[i].Weight * xdata[k - j - 1];

                    x += hiddenNeurons[i].Outputs[0].Weight * Task_Params.ActFunction.Phi(_arg);
                }

                double xe = bias.Outputs[0].Weight;

                for (int i = 0; i < Task_Params.Neurons; i++) {
                    _arg = constant.Outputs[i].Weight;

                    for (int j = 0; j < Task_Params.Dimensions; j++)
                        _arg += inputs[j].Outputs[i].Weight * (xdata[k - j - 1] + dx[j]);

                    xe += hiddenNeurons[i].Outputs[0].Weight * Task_Params.ActFunction.Phi(_arg);
                }

                double rs = 0;

                for (int j = Task_Params.Dimensions - 2; j >= 0; j--) {
                    rs += dx[j] * dx[j];
                    dx[j + 1] = dx[j];
                }

                dx[0] = xe - x;
                rs += dx[0] * dx[0];
                rs = Math.Sqrt(rs / PERTRUB_2);

                for (int j = 0; j < Task_Params.Dimensions; j++)
                    dx[j] /= rs;

                _ltot += Math.Log(rs);
            }

            return _ltot / (nmax - Task_Params.Dimensions);
        }



        private static void ConstructAttractor(double[] xdata, InputNeuron[] inputs, HiddenNeuron[] hiddenNeurons, OutputNeuron outputNeuron, BiasNeuron constant, BiasNeuron bias) {

            long pts = 50000;

            double[] xt = new double[pts];
            double[] yt = new double[pts];
            double[] zt = new double[pts];
            double[] xlast = new double[Task_Params.Dimensions + 1];

            for (int j = 0; j <= Task_Params.Dimensions; j++)
                xlast[j] = xdata[xdata.Length - 1 - j];

            for (long t = 1; t < pts; t++) {
                double xnew = bias.Best[0];

                for (int i = 0; i < Task_Params.Neurons; i++) {
                    double _arg = constant.Best[i];

                    for (int j = 0; j < Task_Params.Dimensions; j++)
                        _arg += inputs[j].Best[i] * xlast[j - 1 + 1];

                    xnew += hiddenNeurons[i].Best[0] * Task_Params.ActFunction.Phi(_arg);
                }

                for (int j = Task_Params.Dimensions; j > 0; j--)
                    xlast[j] = xlast[j - 1];

                yt[t] = xnew;
                xlast[0] = xnew;
                xt[t] = xlast[1];

                if (Task_Params.Dimensions > 2)
                    zt[t] = xlast[2];
                else
                    zt[t] = 1d;
            }

            try {
                double[] constructedSignal = new double[xdata.Length];
                Array.Copy(xt, constructedSignal, xdata.Length);
                PlotObject signal = new SignalPlot(new Timeseries(constructedSignal), new Size(848, 480), 1);
                signal.Plot().Save(NeuralOutput.ReconstructedSignalPlotFileName, ImageFormat.Png);

                PlotObject poincare = new MapPlot(Ext.GeneratePseudoPoincareMapData(xt), new Size(848, 480), 1);
                poincare.Plot().Save(NeuralOutput.ReconstructedPoincarePlotFileName, ImageFormat.Png);
            }
            catch (Exception) { }

            NeuralOutput.Create3dModelFile(xt, yt, zt);

            NeuralOutput.CreateWavFile(yt);
        }



        private static void Prediction(double[] xdata, InputNeuron[] input, HiddenNeuron[] hiddenNeurons, int pointsToPredict, BiasNeuron constant, BiasNeuron bias) {

            if (pointsToPredict == 0)
                return;

            double[] xpred = new double[pointsToPredict + Task_Params.Dimensions];
            double _xpred = 0;

            for (int j = 0; j < Task_Params.Dimensions; j++)
                xpred[Task_Params.Dimensions - j] = xdata[xdata.Length - 1 - j];

            for (int k = Task_Params.Dimensions; k < pointsToPredict + Task_Params.Dimensions; k++) {
                _xpred = bias.Best[0];
                for (int i = 0; i < Task_Params.Neurons; i++) {
                    double _arg = constant.Best[i];

                    for (int j = 0; j < Task_Params.Dimensions; j++)
                        _arg += input[j].Best[i] * xpred[k - j];

                    _xpred += hiddenNeurons[i].Best[0] * Task_Params.ActFunction.Phi(_arg);
                }

                xpred[k] = _xpred;
            }

            PlotObject prediction = new SignalPlot(new Timeseries(xpred), new Size(848, 480), 1);
            prediction.Plot().Save(NeuralOutput.PredictedSignalPlotFileName, ImageFormat.Png);

            NeuralOutput.CreatePredictedDataFile(xpred);
        }



        public Calculations(NeuralNetParams TaskParams) {
            Task_Params = TaskParams;
            PERTRUB = 1e-8;
            PERTRUB_2 = Math.Pow(PERTRUB, 2);
            PERTRUB_SQR_D = PERTRUB / Math.Sqrt(TaskParams.Dimensions);
        }



        public void LoggingEvent() {
            Charts.NeuralAnimation.AddFrame(Charts.DrawNetworkState(800, NeuralNet.NeuronsHidden, NeuralNet.Params.CMax * NeuralNet.successCount + NeuralNet._c));
            Console.WriteLine("{0}\tE: {1:0.#####e-0}", NeuralNet._c, NeuralNet.NeuronOutput.Memory[0]);
        }


        public void EndCycleEvent() {
            double _le = CalculateLargestLyapunovExponent(NeuralNet.xdata, NeuralNet.NeuronsInput, NeuralNet.NeuronsHidden, NeuralNet.NeuronOutput, NeuralNet.NeuronConstant, NeuralNet.NeuronBias);
            Console.WriteLine("\nLLE = {0:F5}\n\n", _le);

            NeuralNet.Task_Result = CalculateLyapunovSpectrum(NeuralNet.xdata, NeuralNet.NeuronsInput, NeuralNet.NeuronsHidden, NeuralNet.System_Equations, NeuralNet.NeuronConstant, NeuralNet.NeuronBias);

            ConstructAttractor(NeuralNet.xdata, NeuralNet.NeuronsInput, NeuralNet.NeuronsHidden, NeuralNet.NeuronOutput, NeuralNet.NeuronConstant, NeuralNet.NeuronBias);
            Prediction(NeuralNet.xdata, NeuralNet.NeuronsInput, NeuralNet.NeuronsHidden, NeuralNet.Params.PtsToPredict, NeuralNet.NeuronConstant, NeuralNet.NeuronBias);

            NeuralOutput.SaveDebugInfoToFile(NeuralNet.NeuronOutput.Memory[0], NeuralNet.Task_Result, _le, NeuralNet.NeuronsInput, NeuralNet.NeuronOutput, NeuralNet.NeuronsHidden, NeuralNet.NeuronConstant, NeuralNet.NeuronBias);

            Charts.DrawNetworkState(1080, NeuralNet.NeuronsHidden, NeuralNet.successCount * Task_Params.CMax).Save(NeuralOutput.NetworkPlotPlotFileName, ImageFormat.Png);
        }

    }
}
