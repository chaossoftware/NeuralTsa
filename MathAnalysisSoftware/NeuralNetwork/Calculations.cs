﻿using MathLib;
using MathLib.DrawEngine;
using MathLib.DrawEngine.Charts;
using MathLib.MathMethods.Lyapunov;
using MathLib.MathMethods.Orthogonalization;
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


        private static BenettinResult CalculateLyapunovSpectrum(double[] xdata, double[,] a, double[] b, NeuralNetEquations systemEquations) {

            int Dim = systemEquations.N;
            int DimPlusOne = systemEquations.NN;

            Orthogonalization Ort = new MGS(Dim);
            BenettinMethod lyap = new BenettinMethod(Dim);
            BenettinResult result;

            double time = 0;                 //time
            int irate = 1;                   //integration steps per reorthonormalization
            int io = xdata.Length - Dim;     //number of iterations of the Map

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
                        x[0, i] = xdata[Dim - i + m];

                    xnew = systemEquations.Derivs(x, a, b);
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


        /// <summary>
        /// Calculate the largest Lyapunov exponent
        /// </summary>
        /// <returns></returns>
        private static double CalculateLargestLyapunovExponent(double[] xdata, double[] dx, double[,] a, double[] b) {
            long nmax = xdata.Length;

            double _arg, x;

            for (int j = 1; j <= Task_Params.Dimensions; j++)
                dx[j] = PERTRUB_SQR_D;

            double _ltot = 0d;

            for (int k = Task_Params.Dimensions + 1; k <= nmax; k++) {
                x = b[0];
                for (int i = 1; i <= Task_Params.Neurons; i++) {
                    _arg = a[i, 0];
                    for (int j = 1; j <= Task_Params.Dimensions; j++)
                        _arg += a[i, j] * xdata[k - j];
                    x += b[i] * Task_Params.ActFunction.Phi(_arg);
                }

                double xe = b[0];

                for (int i = 1; i <= Task_Params.Neurons; i++) {
                    _arg = a[i, 0];
                    for (int j = 1; j <= Task_Params.Dimensions; j++)
                        _arg += a[i, j] * (xdata[k - j] + dx[j]);
                    xe += b[i] * Task_Params.ActFunction.Phi(_arg);
                }

                double rs = 0;

                for (int j = Task_Params.Dimensions - 1; j > 0; j--) {
                    rs += dx[j] * dx[j];
                    dx[j + 1] = dx[j];
                }

                dx[1] = xe - x;
                rs += dx[1] * dx[1];
                rs = Math.Sqrt(rs / PERTRUB_2);

                for (int j = 1; j <= Task_Params.Dimensions; j++)
                    dx[j] /= rs;

                _ltot += Math.Log(rs);
            }

            return _ltot / (nmax - Task_Params.Dimensions);
        }



        private static void ConstructAttractor(double[] xdata, double[] xlast, double[,] averybest, double[] bverybest) {

            long pts = 50000;

            double[] xt = new double[pts];
            double[] yt = new double[pts];
            double[] zt = new double[pts];

            for (int j = 0; j <= Task_Params.Dimensions; j++)
                xlast[j] = xdata[xdata.Length - 1 - j];

            for (long t = 1; t < pts; t++) {
                double xnew = bverybest[0];

                for (int i = 1; i <= Task_Params.Neurons; i++) {
                    double _arg = averybest[i, 0];
                    for (int j = 1; j <= Task_Params.Dimensions; j++)
                        _arg += averybest[i, j] * xlast[j - 1];

                    xnew += bverybest[i] * Task_Params.ActFunction.Phi(_arg);
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
                PlotObject signal = new SignalPlot(new DataSeries(constructedSignal), new Size(848, 480), 1);
                signal.Plot().Save(NeuralOutput.ReconstructedSignalPlotFileName, ImageFormat.Png);

                PlotObject poincare = new MapPlot(Ext.GeneratePseudoPoincareMapData(xt), new Size(848, 480), 1);
                poincare.Plot().Save(NeuralOutput.ReconstructedPoincarePlotFileName, ImageFormat.Png);
            }
            catch (Exception) { }

            NeuralOutput.Create3dModelFile(xt, yt, zt);

            NeuralOutput.CreateWavFile(yt);
        }



        private static void Prediction(double[] xdata, double[,] averybest, double[] bverybest, int pointsToPredict) {

            if (pointsToPredict == 0)
                return;

            double[] xpred = new double[pointsToPredict + Task_Params.Dimensions + 1];
            double _xpred = 0;

            for (int j = 0; j <= Task_Params.Dimensions; j++)
                xpred[Task_Params.Dimensions - j] = xdata[xdata.Length - 1 - j];

            for (int k = Task_Params.Dimensions + 1; k <= pointsToPredict + Task_Params.Dimensions; k++) {
                _xpred = bverybest[0];
                for (int i = 1; i <= Task_Params.Neurons; i++) {
                    double _arg = averybest[i, 0];

                    for (int j = 1; j <= Task_Params.Dimensions; j++)
                        _arg += averybest[i, j] * xpred[k - j];

                    _xpred += bverybest[i] * Task_Params.ActFunction.Phi(_arg);
                }

                xpred[k] = _xpred;
            }

            PlotObject prediction = new SignalPlot(new DataSeries(xpred), new Size(848, 480), 1);
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
            Charts.AddFrame(Charts.DrawNetworkState(800, NeuralNet.a, NeuralNet.b, NeuralNet.Task_Params.Dimensions, NeuralNet.Task_Params.Neurons, NeuralNet.Task_Params.CMax * NeuralNet.successCount + NeuralNet._c));
            Console.WriteLine("{0}\tE: {1:0.#####e-0}", NeuralNet._c, NeuralNet.ebest);
        }


        public void EndCycleEvent() {
            double _le = CalculateLargestLyapunovExponent(NeuralNet.xdata, NeuralNet.dx, NeuralNet.a, NeuralNet.b);
            Console.WriteLine("\nLLE = {0:F5}\n\n", _le);

            NeuralNet.Task_Result = CalculateLyapunovSpectrum(NeuralNet.xdata, NeuralNet.a, NeuralNet.b, NeuralNet.System_Equations);

            ConstructAttractor(NeuralNet.xdata, NeuralNet.xlast, NeuralNet.averybest, NeuralNet.bverybest);
            Prediction(NeuralNet.xdata, NeuralNet.averybest, NeuralNet.bverybest, NeuralNet.Task_Params.PtsToPredict);

            NeuralOutput.SaveDebugInfoToFile(NeuralNet.ebest, NeuralNet.Task_Result, _le, NeuralNet.bbest, NeuralNet.abest, Task_Params.Neurons, Task_Params.Dimensions);

            Charts.DrawNetworkState(1080, NeuralNet.a, NeuralNet.b, Task_Params.Dimensions, Task_Params.Neurons, NeuralNet.successCount * Task_Params.CMax).Save(NeuralOutput.NetworkPlotPlotFileName, ImageFormat.Png);
        }

    }
}
