using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Globalization;
using System.Text;
using MathLib;
using MathLib.Data;
using MathLib.DrawEngine;
using MathLib.DrawEngine.Charts;
using MathLib.IO;
using MathLib.MathMethods.Lyapunov;
using MathLib.MathMethods.Orthogonalization;
using MathLib.Transform;
using NeuralAnalyser.Configuration;
using NeuralAnalyser.NeuralNet;
using NeuralAnalyser.NeuralNet.Entities;

namespace NeuralAnalyser
{
    internal class Calculations
    {
        private const double Perturbation = 1e-8; //Perturbation size

        private double perturbationDivSqrtD;
        private double perturbationSqr; //Pertrubation^2
        private NeuralNetParameters netParams;
        private OutputParameters outParams;
        private BenettinResult benettin;
        private NeuralNetEquations systemEquations;

        private List<double> errors = new List<double>() { 1 };

        public Visualizer Visualizator { get; set; }

        public Calculations(NeuralNetParameters parameters, OutputParameters outParameters)
        {
            this.netParams = parameters;
            this.outParams = outParameters;

            this.Visualizator = new Visualizer(outParameters.AnimationSize);

            if (outParams.SaveAnimation)
            {
                this.Visualizator.NeuralAnimation = new Animation();
            }

            perturbationSqr = Math.Pow(Perturbation, 2);
            perturbationDivSqrtD = Perturbation / Math.Sqrt(parameters.Dimensions);
            systemEquations = new NeuralNetEquations(parameters.Dimensions, parameters.Neurons, parameters.ActFunction);
            
        }

        public void LogCycle(SciNeuralNet net)
        {
            if (outParams.SaveAnimation)
            {
                this.Visualizator.NeuralAnimation.AddFrame(PrepareAnimationFrame(net));
                //this.Visualizator.NeuralAnimation.AddFrame(Visualizator.DrawBrain(net));
            }

            Console.WriteLine("{0}\tE: {1:0.#####e-0}", net.current, net.OutputLayer.Neurons[0].Memory[0]);
        }

        public void PerformCalculations(SciNeuralNet net)
        {
            double lle = CalculateLargestLyapunovExponent(net);
            Console.WriteLine("----------------------------------");
            Console.WriteLine("Epoch {0}", net.successCount);
            Console.WriteLine("LLE = {0:F5}", lle);
            Console.WriteLine("----------------------------------\n\n");


            benettin = CalculateLyapunovSpectrum(net, systemEquations);

            ConstructAttractor(net);

            if (net.Params.PtsToPredict > 0)
            {
                Prediction(net);
            }

            SaveDebugInfoToFile(net, benettin, lle);

            Visualizator.DrawBrain(net).Save(outParams.NetworkPlotPlotFileName, ImageFormat.Png);
        }

        private Bitmap PrepareAnimationFrame(SciNeuralNet net)
        {
            errors.Add(Math.Min(net.OutputLayer.Neurons[0].Memory[0], 1));

            var result = new Bitmap(outParams.AnimationSize.Width * 2, outParams.AnimationSize.Height);
            var netImg = Visualizator.DrawBrain(net);

            var plot = new MultiSignalPlot(outParams.AnimationSize, 1);
            plot.AddDataSeries(new Timeseries(new double[] { 0, 0 }), Color.Black);
            plot.AddDataSeries(new Timeseries(new double[] { 1, 1 }), Color.Black);
            plot.AddDataSeries(new Timeseries(errors.ToArray()), Color.Blue);
            var chart = plot.Plot();

            var stringFormat = new StringFormat();
            stringFormat.Alignment = StringAlignment.Far;
            //stringFormat.LineAlignment = StringAlignment.Center;
            var font = new Font(new FontFamily("Cambria Math"), 13f);
            var textBrush = new SolidBrush(Color.Black);

            using (Graphics g = Graphics.FromImage(result))
            {
                g.TextRenderingHint = TextRenderingHint.AntiAlias;
                g.DrawImage(netImg, new Point(0, 0));
                g.DrawImage(chart, new Point(outParams.AnimationSize.Width, 0));

                g.DrawString(string.Format("Dimensions: {1}\nNeurons: {0}\nIteration: {2:N0}", net.Params.Neurons, net.Params.Dimensions, net.current + net.successCount * net.Params.CMax), font, textBrush, outParams.AnimationSize.Width * 2, 0f, stringFormat);
            }

            return result;
        }

        private BenettinResult CalculateLyapunovSpectrum(SciNeuralNet net, NeuralNetEquations systemEquations)
        {

            int Dim = systemEquations.EquationsCount;
            int DimPlusOne = systemEquations.TotalEquationsCount;

            Orthogonalization Ort = new ModifiedGrammSchmidt(Dim);
            BenettinMethod lyap = new BenettinMethod(Dim);
            BenettinResult result;

            double time = 0;                 //time
            int irate = 1;                   //integration steps per reorthonormalization
            int io = net.xdata.Length - Dim - 1;     //number of iterations of the Map

            double[,] x = new double[DimPlusOne, Dim];
            double[,] xnew = new double[DimPlusOne, Dim];
            double[,] v = new double[DimPlusOne, Dim];
            double[] znorm = new double[Dim];

            double[,] leInTime = new double[Dim, io];

            systemEquations.Init(v, net.xdata);

            for (int m = 0; m < io; m++) {
                for (int j = 0; j < irate; j++) {

                    Array.Copy(v, x, v.Length);

                    //Use actual data rather than iterated data
                    for (int i = 0; i < Dim; i++)
                        x[0, i] = net.xdata[Dim - i + m - 1];

                    xnew = systemEquations.Derivs(x, Get2DArray(net.HiddenLayer.Neurons, net.InputLayer.Neurons, net.NeuronConstant), Get1DArray(net.HiddenLayer.Neurons), net.NeuronBias.Outputs[0].Weight, net.NeuronConstant);
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

            if (outParams.SaveLeInTime)
            {
                CreateLeInTimeFile(leInTime);
            }

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
            double[] dx = new double[netParams.Dimensions];

            for (int j = 0; j < netParams.Dimensions; j++)
                dx[j] = perturbationDivSqrtD;

            double _ltot = 0d;

            for (int k = netParams.Dimensions; k < nmax; k++)
            {
                x = net.NeuronBias.Outputs[0].Weight;
                for (int i = 0; i < netParams.Neurons; i++)
                {
                    _arg = net.NeuronConstant.Outputs[i].Weight;

                    for (int j = 0; j < netParams.Dimensions; j++)
                        _arg += net.InputLayer.Neurons[j].Outputs[i].Weight * net.xdata[k - j - 1];

                    x += net.HiddenLayer.Neurons[i].Outputs[0].Weight * netParams.ActFunction.Phi(_arg);
                }

                double xe = net.NeuronBias.Outputs[0].Weight;

                for (int i = 0; i < netParams.Neurons; i++)
                {
                    _arg = net.NeuronConstant.Outputs[i].Weight;

                    for (int j = 0; j < netParams.Dimensions; j++)
                        _arg += net.InputLayer.Neurons[j].Outputs[i].Weight * (net.xdata[k - j - 1] + dx[j]);

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
                    dx[j] /= rs;

                _ltot += Math.Log(rs);
            }

            return _ltot / (nmax - netParams.Dimensions);
        }

        private void ConstructAttractor(SciNeuralNet net)
        {
            long pts = 50000;

            double[] xt = new double[pts];
            double[] yt = new double[pts];
            double[] zt = new double[pts];
            double[] xlast = new double[netParams.Dimensions + 1];

            for (int j = 0; j <= netParams.Dimensions; j++)
                xlast[j] = net.xdata[net.xdata.Length - 1 - j];

            for (long t = 1; t < pts; t++)
            {
                double xnew = net.NeuronBias.Best[0];

                for (int i = 0; i < netParams.Neurons; i++)
                {
                    double _arg = net.NeuronConstant.Best[i];

                    for (int j = 0; j < netParams.Dimensions; j++)
                        _arg += net.InputLayer.Neurons[j].Best[i] * xlast[j - 1 + 1];

                    xnew += net.HiddenLayer.Neurons[i].Best[0] * netParams.ActFunction.Phi(_arg);
                }

                for (int j = netParams.Dimensions; j > 0; j--)
                    xlast[j] = xlast[j - 1];

                yt[t] = xnew;
                xlast[0] = xnew;
                xt[t] = xlast[1];

                zt[t] = netParams.Dimensions > 2 ? xlast[2] : 1d;
            }

            try
            {
                double[] constructedSignal = new double[net.xdata.Length];
                Array.Copy(xt, constructedSignal, net.xdata.Length);

                new SignalPlot(new Timeseries(constructedSignal), outParams.PlotsSize, 1)
                    .Plot()
                    .Save(outParams.ReconstructedSignalPlotFileName, ImageFormat.Png);

                new MapPlot(Ext.GeneratePseudoPoincareMapData(xt), outParams.PlotsSize, 1)
                    .Plot()
                    .Save(outParams.ReconstructedPoincarePlotFileName, ImageFormat.Png);
            }
            catch
            {
                Console.WriteLine("Prediction was not succeeded, unable to build charts");
            }

            if (outParams.SaveModel)
            {
                Model3D.Create3dPlyModelFile(outParams.ModelFileName, xt, yt, zt);
            }

            if (outParams.SaveWav)
            {
                Sound.CreateWavFile(outParams.WavFileName, yt);
            }
        }

        private void Prediction(SciNeuralNet net) {

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

            var prediction = new SignalPlot(new Timeseries(xpred), outParams.PlotsSize, 1);
            prediction.Plot().Save(outParams.PredictedSignalPlotFileName, ImageFormat.Png);

            var pred = new StringBuilder();

            foreach (double predictedPoint in xpred)
            {
                pred.AppendFormat(CultureInfo.InvariantCulture, "{0:" + NumFormat.General + "}\n", predictedPoint);
            }

            DataWriter.CreateDataFile(outParams.PredictFileName, prediction.ToString());
        }

        private void SaveDebugInfoToFile(SciNeuralNet net, BenettinResult benettin, double lle)
        {
            var ebest = net.OutputLayer.Neurons[0].Memory[0];

            var debug = new StringBuilder()
                .AppendFormat(CultureInfo.InvariantCulture, "Training error: {0:0.#####e-0}\n\n", ebest)
                .Append(benettin.GetInfo())
                .AppendFormat(CultureInfo.InvariantCulture, "Largest Lyapunov exponent: {0:F5}\n", lle)
                .AppendFormat(CultureInfo.InvariantCulture, "\nBias: {0:F8}\n\n", net.NeuronBias.Memory[0]);

            for (int i = 0; i < net.Params.Neurons; i++)
            {
                debug.AppendFormat("Neuron {0} :\t\t\t", i + 1);
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

            DataWriter.CreateDataFile(outParams.LeInTimeFileName, le.ToString());
        }
    }
}
