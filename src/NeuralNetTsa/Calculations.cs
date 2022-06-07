﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Windows.Forms.DataVisualization.Charting;
using ChaosSoft.Core;
using ChaosSoft.Core.Data;
using ChaosSoft.Core.DrawEngine;
using ChaosSoft.Core.DrawEngine.Charts;
using ChaosSoft.Core.IO;
using ChaosSoft.Core.NumericalMethods.Lyapunov;
using ChaosSoft.Core.NumericalMethods.Orthogonalization;
using ChaosSoft.Core.Transform;
using NeuralNetTsa.Configuration;
using NeuralNetTsa.NeuralNet;
using NeuralNetTsa.NeuralNet.Entities;

namespace NeuralNetTsa
{
    internal class Calculations
    {
        private const double Perturbation = 1e-8; //Perturbation size

        private readonly double perturbationDivSqrtD;
        private readonly double perturbationSqr; //Pertrubation^2
        private readonly NeuralNetParameters netParams;
        private readonly OutputParameters outParams;
        private readonly NeuralNetEquations systemEquations;

        private readonly List<double[]> errorsHistory = new List<double[]>();

        private readonly List<double> errors = new List<double>();

        private readonly Size _squareSize;
        private readonly Size _rectangleSize;


        private Bitmap poincare = null;
        private Bitmap signalOriginal = null;
        private Bitmap signal = null;

        public Visualizer Visualizator { get; set; }

        public Calculations(NeuralNetParameters parameters, OutputParameters outParameters)
        {
            netParams = parameters;
            outParams = outParameters;

            _squareSize = new Size(outParams.AnimationSize.Width / 2, outParams.AnimationSize.Height / 2);
            _rectangleSize = new Size(outParams.AnimationSize.Width, outParams.AnimationSize.Height / 4);

            var netImageSize = new Size(outParams.AnimationSize.Width / 8, outParams.AnimationSize.Height / 4);
            Visualizator = new Visualizer(netImageSize);

            if (outParams.SaveAnimation)
            {
                Visualizator.NeuralAnimation = new Animation();
            }

            perturbationSqr = Math.Pow(Perturbation, 2);
            perturbationDivSqrtD = Perturbation / Math.Sqrt(parameters.Dimensions);
            systemEquations = new NeuralNetEquations(parameters.Dimensions, parameters.Neurons, parameters.ActFunction);
        }

        public void LogCycle(SciNeuralNet net)
        {
            if (outParams.SaveAnimation)
            {
                Visualizator.NeuralAnimation.AddFrame(PrepareAnimationFrame(net));
            }

            Console.SetCursorPosition(0, 19);
            Console.WriteLine(net.current.ToString().PadRight(10) + "err: {0:e}\t\t", net.OutputLayer.Neurons[0].Memory[0]);
            Console.WriteLine(new string('-', 40));
            int offset = ConsoleNetVisualizer.Visualize(net, 20);
            Console.SetCursorPosition(0, offset);
            Console.WriteLine(new string('-', 40));
        }

        public void PerformCalculations(SciNeuralNet net)
        {
            double lle = CalculateLargestLyapunovExponent(net);
            
            Console.WriteLine("Epoch {0}", net.successCount);
            Console.WriteLine("LLE = {0:F5}", lle);

            var benettin = CalculateLyapunovSpectrum(net, systemEquations);

            ConstructAttractor(net);

            if (net.Params.PtsToPredict > 0)
            {
                Prediction(net);
            }

            SaveDebugInfoToFile(net, benettin, lle);

            Visualizator.DrawBrain(net).Save(outParams.NetPlotFile, ImageFormat.Png);

            if (outParams.SaveAnimation)
            {
                errorsHistory.Add(errors.ToArray());
                errors.Clear();
            }
        }

        private Bitmap PrepareAnimationFrame(SciNeuralNet net)
        {
            if (poincare == null)
            {
                var pPlot = new ScatterPlot(_squareSize);
                pPlot.AddDataSeries(PseudoPoincareMap.GetMapDataFrom(net.xdata), Color.OrangeRed, 1.5f);

                poincare = pPlot.Plot();
            }

            if (signalOriginal == null)
            {
                var sPlot = new LinePlot(_rectangleSize);
                sPlot.AddDataSeries(new Timeseries(net.xdata), Color.OrangeRed);
                signalOriginal = sPlot.Plot();

                var sPlot1 = new LinePlot(_rectangleSize);
                signal = sPlot1.Plot();
            }

            var error = Math.Log10(net.OutputLayer.Neurons[0].Memory[0]);
            error = FastMath.Min(error, 0d);
            error = FastMath.Max(error, -10d);

            if (!errors.Any())
            {
                errors.Add(error);
            }

            errors.Add(error);

            var result = new Bitmap(outParams.AnimationSize.Width, outParams.AnimationSize.Height);
            var netImg = Visualizator.DrawBrain(net);

            var plot = new LinePlot(new Size(outParams.AnimationSize.Width / 2, outParams.AnimationSize.Height / 2));

            plot.AddDataSeries(new Timeseries(new double[] { 0, 0 }), Color.Black);
            plot.AddDataSeries(new Timeseries(new double[] { -10, -10 }), Color.Black);

            foreach (var tSeries in errorsHistory)
            {
                plot.AddDataSeries(new Timeseries(tSeries), Color.LightBlue);
            }

            plot.AddDataSeries(new Timeseries(errors.ToArray()), Color.Blue);

            plot.LabelY = "Training error (Log10)";
            plot.LabelX = "cycle #";
            var chart = plot.Plot();

            var stringFormat = new StringFormat();
            stringFormat.Alignment = StringAlignment.Far;
            var font = new Font(new FontFamily("Cambria Math"), 11f);
            var textBrush = new SolidBrush(Color.Black);

            using (Graphics g = Graphics.FromImage(result))
            {
                g.TextRenderingHint = TextRenderingHint.AntiAlias;
                g.DrawImage(poincare, new Point(0, 0));
                g.DrawImage(chart, new Point(outParams.AnimationSize.Width / 2, 0));
                g.DrawImage(netImg, new Point(outParams.AnimationSize.Width - netImg.Width, 0));
                g.DrawImage(signalOriginal, new Point(0, outParams.AnimationSize.Height / 2));
                g.DrawImage(signal, new Point(0, outParams.AnimationSize.Height / 4 * 3));

                g.DrawString(string.Format("Dimensions: {1}\nNeurons: {0}\nIteration: {2:N0}", net.Params.Neurons, net.Params.Dimensions, net.current + net.successCount * net.Params.EpochInterval), font, textBrush, outParams.AnimationSize.Width * 2, 0f, stringFormat);
            }

            return result;
        }

        private LyapunovSpectrum CalculateLyapunovSpectrum(SciNeuralNet net, NeuralNetEquations systemEquations)
        {
            int dim = systemEquations.EquationsCount;
            int dimPlusOne = systemEquations.TotalEquationsCount;

            var ort = new ModifiedGrammSchmidt(dim);
            var lyap = new BenettinMethod(dim);

            double time = 0;                 //time
            int irate = 1;                   //integration steps per reorthonormalization
            int io = net.xdata.Length - dim - 1;     //number of iterations of the Map

            double[,] x = new double[dimPlusOne, dim];
            double[,] xnew;// = new double[DimPlusOne, Dim];
            double[,] v = new double[dimPlusOne, dim];
            double[] znorm = new double[dim];

            double[,] leInTime = new double[dim, io];

            systemEquations.Init(v, net.xdata);

            for (int m = 0; m < io; m++)
            {
                for (int j = 0; j < irate; j++)
                {
                    Array.Copy(v, x, v.Length);

                    //Use actual data rather than iterated data
                    for (int i = 0; i < dim; i++)
                    {
                        x[0, i] = net.xdata[dim - i + m - 1];
                    }

                    xnew = systemEquations.Derivs(x, Get2DArray(net.HiddenLayer.Neurons, net.InputLayer.Neurons), Get1DArray(net.HiddenLayer.Neurons), net.NeuronBias.Outputs[0].Weight, net.NeuronConstant);
                    Array.Copy(xnew, v, xnew.Length);

                    time++;
                }

                ort.Perform(v, znorm);
                lyap.CalculateLyapunovSpectrum(znorm, time);

                for (int k = 0; k < dim; k++)
                {
                    if (znorm[k] > 0)
                    {
                        leInTime[k, m] = Math.Log(znorm[k]);
                    }
                }
            }

            lyap.Result.SpectrumInTime = leInTime;

            if (outParams.SaveLeInTime)
            {
                CreateLeInTimeFile(leInTime);
            }

            return lyap.Result;
        }

        private double[,] Get2DArray(HiddenNeuron[] neurons, InputNeuron[] inputs)
        {
            double[,] arr = new double[neurons.Length, inputs.Length];

            for (int i = 0; i < neurons.Length; i++)
            {
                for (int j = 0; j < inputs.Length; j++)
                {
                    arr[i, j] = inputs[j].Outputs[i].Weight;

                }
            }

            return arr;
        }

        private double[] Get1DArray(HiddenNeuron[] neurons)
        {
            double[] arr = new double[neurons.Length];

            for (int i = 0; i < neurons.Length; i++)
            {
                arr[i] = neurons[i].Outputs[0].Weight;
            }

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
            double[] dx = new double[netParams.Dimensions];
            double ltot = 0d;

            for (int j = 0; j < netParams.Dimensions; j++)
            {
                dx[j] = perturbationDivSqrtD;
            }

            for (int k = netParams.Dimensions; k < nmax; k++)
            {
                x = net.NeuronBias.Outputs[0].Weight;

                for (int i = 0; i < netParams.Neurons; i++)
                {
                    _arg = net.NeuronConstant.Outputs[i].Weight;

                    for (int j = 0; j < netParams.Dimensions; j++)
                    {
                        _arg += net.InputLayer.Neurons[j].Outputs[i].Weight * net.xdata[k - j - 1];
                    }

                    x += net.HiddenLayer.Neurons[i].Outputs[0].Weight * netParams.ActFunction.Phi(_arg);
                }

                double xe = net.NeuronBias.Outputs[0].Weight;

                for (int i = 0; i < netParams.Neurons; i++)
                {
                    _arg = net.NeuronConstant.Outputs[i].Weight;

                    for (int j = 0; j < netParams.Dimensions; j++)
                    {
                        _arg += net.InputLayer.Neurons[j].Outputs[i].Weight * (net.xdata[k - j - 1] + dx[j]);
                    }

                    xe += net.HiddenLayer.Neurons[i].Outputs[0].Weight * netParams.ActFunction.Phi(_arg);
                }

                double rs = 0;

                for (int j = netParams.Dimensions - 2; j >= 0; j--)
                {
                    rs += dx[j] * dx[j];
                    dx[j + 1] = dx[j];
                }

                dx[0] = xe - x;
                rs += dx[0] * dx[0];
                rs = Math.Sqrt(rs / perturbationSqr);

                for (int j = 0; j < netParams.Dimensions; j++)
                {
                    dx[j] /= rs;
                }

                ltot += Math.Log(rs);
            }

            return ltot / (nmax - netParams.Dimensions);
        }

        private void ConstructAttractor(SciNeuralNet net)
        {
            long pts = outParams.PredictedSignalPts;

            double[] xt = new double[pts];
            double[] yt = new double[pts];
            double[] zt = new double[pts];
            double[] xlast = new double[netParams.Dimensions + 1];

            for (int j = 0; j <= netParams.Dimensions; j++)
            {
                xlast[j] = net.xdata[net.xdata.Length - 1 - j];
            }

            for (long t = 1; t < pts; t++)
            {
                double xnew = net.NeuronBias.Best[0];

                for (int i = 0; i < netParams.Neurons; i++)
                {
                    double _arg = net.NeuronConstant.Best[i];

                    for (int j = 0; j < netParams.Dimensions; j++)
                    {
                        _arg += net.InputLayer.Neurons[j].Best[i] * xlast[j - 1 + 1];
                    }

                    xnew += net.HiddenLayer.Neurons[i].Best[0] * netParams.ActFunction.Phi(_arg);
                }

                for (int j = netParams.Dimensions; j > 0; j--)
                {
                    xlast[j] = xlast[j - 1];
                }

                yt[t] = xnew;
                xlast[0] = xnew;
                xt[t] = xlast[1];

                zt[t] = netParams.Dimensions > 2 ? xlast[2] : 1d;
            }

            try
            {
                new MathChart(outParams.PlotsSize, "t", "f(t)")
                    .AddTimeSeries("Signal", new Timeseries(xt), SeriesChartType.Line)
                    .SaveImage(outParams.ReconstructedSignalPlotFile, ImageFormat.Png);

                new MathChart(outParams.PlotsSize, "t", "t + 1")
                    .AddTimeSeries("Pseudo poincare", PseudoPoincareMap.GetMapDataFrom(xt), SeriesChartType.Point)
                    .SaveImage(outParams.ReconstructedPoincarePlotFile, ImageFormat.Png);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Prediction was not succeeded, unable to build charts: " + ex);
            }

            if (outParams.SaveModel)
            {
                Model3D.Create3daModelFile(outParams.ModelFile, xt, yt, zt);
            }

            if (outParams.SaveWav)
            {
                Sound.CreateWavFile(outParams.WavFile, yt);
            }

            if (outParams.SaveAnimation)
            {
                try
                {
                    var pPlot = new ScatterPlot(_squareSize);
                    pPlot.AddDataSeries(PseudoPoincareMap.GetMapDataFrom(xt), Color.SteelBlue);
                    pPlot.AddDataSeries(PseudoPoincareMap.GetMapDataFrom(net.xdata), Color.OrangeRed, 1.5f);
                    poincare = pPlot.Plot();
                }
                catch
                {
                    var pPlot = new ScatterPlot(_squareSize);
                    pPlot.AddDataSeries(PseudoPoincareMap.GetMapDataFrom(net.xdata), Color.OrangeRed, 1.5f);
                    poincare = pPlot.Plot();
                }

                try
                {
                    var sPlot = new LinePlot(_rectangleSize);
                    sPlot.AddDataSeries(new Timeseries(xt.Take(net.xdata.Length).ToArray()), Color.SteelBlue);
                    signal = sPlot.Plot();
                }
                catch 
                { 
                }
            }
        }

        private void Prediction(SciNeuralNet net)
        {
            var pointsToPredict = net.Params.PtsToPredict;

            double[] xpred = new double[pointsToPredict + netParams.Dimensions];
            double _xpred = 0;

            for (int j = 0; j < netParams.Dimensions; j++)
            {
                xpred[netParams.Dimensions - j] = net.xdata[net.xdata.Length - 1 - j];
            }

            for (int k = netParams.Dimensions; k < pointsToPredict + netParams.Dimensions; k++)
            {
                _xpred = net.NeuronBias.Best[0];

                for (int i = 0; i < netParams.Neurons; i++)
                {
                    double _arg = net.NeuronConstant.Best[i];

                    for (int j = 0; j < netParams.Dimensions; j++)
                    {
                        _arg += net.InputLayer.Neurons[j].Best[i] * xpred[k - j];
                    }

                    _xpred += net.HiddenLayer.Neurons[i].Best[0] * netParams.ActFunction.Phi(_arg);
                }

                xpred[k] = _xpred;
            }

            new MathChart(outParams.PlotsSize, "t", "f(t)")
                .AddTimeSeries("Signal", new Timeseries(xpred), SeriesChartType.Line)
                .SaveImage(outParams.PredictedSignalPlotFile, ImageFormat.Png);

            var pred = new StringBuilder();

            foreach (double predictedPoint in xpred)
            {
                pred.AppendLine(NumFormat.ToLong(predictedPoint));
            }

            DataWriter.CreateDataFile(outParams.PredictFile, pred.ToString());
        }

        private void SaveDebugInfoToFile(SciNeuralNet net, LyapunovSpectrum benettin, double lle)
        {
            var ebest = net.OutputLayer.Neurons[0].Memory[0];

            var debug = new StringBuilder()
                .AppendFormat(CultureInfo.InvariantCulture, "Training error: {0:0.#####e-0}\n\n", ebest)
                .Append(benettin.ToString())
                .AppendFormat(CultureInfo.InvariantCulture, "Largest Lyapunov exponent: {0:F5}\n", lle)
                .AppendFormat(CultureInfo.InvariantCulture, "\nBias: {0:F8}\n\n", net.NeuronBias.Memory[0]);

            for (int i = 0; i < net.Params.Neurons; i++)
            {
                debug.AppendFormat("Neuron {0} :\t\t", i + 1);
            }

            debug.AppendLine();

            foreach (var memory in net.NeuronConstant.Memory)
            {
                debug.AppendFormat(CultureInfo.InvariantCulture, "{0:F8}\t\t", memory);
            }

            debug.AppendLine();

            foreach (var neuron in net.InputLayer.Neurons)
            {
                foreach (var memory in neuron.Memory)
                {
                    debug.AppendFormat(CultureInfo.InvariantCulture, "{0:F8}\t\t", memory);
                }

                debug.AppendLine();
            }

            debug.AppendLine();

            foreach (var neuron in net.HiddenLayer.Neurons)
            {
                debug.AppendFormat(CultureInfo.InvariantCulture, "{0:F8}\t\t", neuron.Memory[0]);
            }

            debug.AppendLine().AppendLine(new string('-', 50));

            Logger.LogInfo(debug.ToString(), true);
        }

        private void CreateLeInTimeFile(double[,] leInTime)
        {
            var le = new StringBuilder();

            for (int i = 0; i < leInTime.GetLength(1); i++)
            {
                for (int j = 0; j < leInTime.GetLength(0); j++)
                {
                    le.AppendFormat(CultureInfo.InvariantCulture, "{0:G5}\t", leInTime[j, i]);
                }

                le.AppendLine();
            }

            DataWriter.CreateDataFile(outParams.LeInTimeFile, le.ToString());
        }
    }
}