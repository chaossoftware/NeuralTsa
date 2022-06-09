using System;
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
using ChaosSoft.Core.Transform;
using NeuralNetTsa.Configuration;
using NeuralNetTsa.NeuralNet;
using NeuralNetTsa.NeuralNet.Entities;
using NeuralNetTsa.Routines;

namespace NeuralNetTsa
{
    internal class Calculations
    {
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
            outParams = outParameters;

            _squareSize = new Size(outParams.AnimationSize.Width / 2, outParams.AnimationSize.Height / 2);
            _rectangleSize = new Size(outParams.AnimationSize.Width, outParams.AnimationSize.Height / 4);

            Size netImageSize = new Size(outParams.AnimationSize.Width / 8, outParams.AnimationSize.Height / 4);
            Visualizator = new Visualizer(netImageSize, outParams.SaveAnimation);

            systemEquations = new NeuralNetEquations(parameters.Dimensions, parameters.Neurons, parameters.ActFunction);
        }

        public void AddAnimationFrame(SciNeuralNet net)
        {
            if (outParams.SaveAnimation)
            {
                Visualizator.NeuralAnimation.AddFrame(PrepareAnimationFrame(net));
            }
        }

        public void PerformCalculations(SciNeuralNet net)
        {
            double lle = Lle.Calculate(net);
            
            Console.WriteLine("Epoch {0}\nLLE = {1:F5}", net.successCount, lle);

            LyapunovSpectrum leSpec = LeSpec.Calculate(net, systemEquations);

            if (outParams.SaveLeInTime)
            {
                DataWriter.CreateDataFile(outParams.LeInTimeFile, leSpec.SpectrumInTime);
            }

            AttractorData data = Attractor.Construct(net, outParams.PredictedSignalPts);

            try
            {
                new MathChart(outParams.PlotsSize, "t", "f(t)")
                    .AddTimeSeries("Signal", new Timeseries(data.Xt), SeriesChartType.Line)
                    .SaveImage(outParams.ReconstructedSignalPlotFile, ImageFormat.Png);

                new MathChart(outParams.PlotsSize, "t", "t + 1")
                    .AddTimeSeries("Pseudo poincare", PseudoPoincareMap.GetMapDataFrom(data.Xt), SeriesChartType.Point)
                    .SaveImage(outParams.ReconstructedPoincarePlotFile, ImageFormat.Png);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Prediction was not succeeded, unable to build charts: " + ex);
            }

            if (outParams.SaveModel)
            {
                Model3D.Create3daModelFile(outParams.ModelFile, data.Xt, data.Yt, data.Zt);
            }

            if (outParams.SaveWav)
            {
                Sound.CreateWavFile(outParams.WavFile, data.Yt);
            }

            if (outParams.SaveAnimation)
            {
                try
                {
                    ScatterPlot pPlot = new ScatterPlot(_squareSize);
                    pPlot.AddDataSeries(PseudoPoincareMap.GetMapDataFrom(data.Xt), Color.SteelBlue);
                    pPlot.AddDataSeries(PseudoPoincareMap.GetMapDataFrom(net.xdata), Color.OrangeRed, 1.5f);
                    poincare = pPlot.Plot();
                }
                catch
                {
                    ScatterPlot pPlot = new ScatterPlot(_squareSize);
                    pPlot.AddDataSeries(PseudoPoincareMap.GetMapDataFrom(net.xdata), Color.OrangeRed, 1.5f);
                    poincare = pPlot.Plot();
                }

                try
                {
                    LinePlot sPlot = new LinePlot(_rectangleSize);
                    sPlot.AddDataSeries(new Timeseries(data.Xt.Take(net.xdata.Length).ToArray()), Color.SteelBlue);
                    signal = sPlot.Plot();
                }
                catch
                {

                }
            }

            if (net.Params.PtsToPredict > 0)
            {
                Prediction(net);
            }

            SaveDebugInfoToFile(net, leSpec, lle);

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
                ScatterPlot pPlot = new ScatterPlot(_squareSize);
                pPlot.AddDataSeries(PseudoPoincareMap.GetMapDataFrom(net.xdata), Color.OrangeRed, 1.5f);

                poincare = pPlot.Plot();
            }

            if (signalOriginal == null)
            {
                LinePlot sPlot = new LinePlot(_rectangleSize);
                sPlot.AddDataSeries(new Timeseries(net.xdata), Color.OrangeRed);
                signalOriginal = sPlot.Plot();

                LinePlot sPlot1 = new LinePlot(_rectangleSize);
                signal = sPlot1.Plot();
            }

            double error = Math.Log10(net.OutputLayer.Neurons[0].Memory[0]);
            error = FastMath.Min(error, 0);
            error = FastMath.Max(error, -10);

            if (!errors.Any())
            {
                errors.Add(error);
            }

            errors.Add(error);

            Bitmap result = new Bitmap(outParams.AnimationSize.Width, outParams.AnimationSize.Height);
            Bitmap netImg = Visualizator.DrawBrain(net);

            LinePlot plot = new LinePlot(new Size(outParams.AnimationSize.Width / 2, outParams.AnimationSize.Height / 2));

            plot.AddDataSeries(new Timeseries(new double[] { 0, 0 }), Color.Black);
            plot.AddDataSeries(new Timeseries(new double[] { -10, -10 }), Color.Black);

            foreach (var tSeries in errorsHistory)
            {
                plot.AddDataSeries(new Timeseries(tSeries), Color.LightBlue);
            }

            plot.AddDataSeries(new Timeseries(errors.ToArray()), Color.Blue);

            plot.LabelY = "Training error (Log10)";
            plot.LabelX = "cycle #";
            Bitmap chart = plot.Plot();

            StringFormat stringFormat = new StringFormat
            {
                Alignment = StringAlignment.Far
            };

            Font font = new Font(new FontFamily("Cambria Math"), 11f);
            SolidBrush textBrush = new SolidBrush(Color.Black);

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

        private void Prediction(SciNeuralNet net)
        {
            double[] xPredicted = SignalPrediction.Make(net, net.Params.PtsToPredict);

            StringBuilder prediction = new StringBuilder();
            Array.ForEach(xPredicted, x => prediction.AppendLine(NumFormat.ToLong(x)));
            DataWriter.CreateDataFile(outParams.PredictFile, prediction.ToString());

            new MathChart(outParams.PlotsSize, "t", "f(t)")
                .AddTimeSeries("Signal", new Timeseries(xPredicted), SeriesChartType.Line)
                .SaveImage(outParams.PredictedSignalPlotFile, ImageFormat.Png);
        }

        private void SaveDebugInfoToFile(SciNeuralNet net, LyapunovSpectrum les, double lle)
        {
            double eBest = net.OutputLayer.Neurons[0].Memory[0];

            StringBuilder debug = new StringBuilder()
                .AppendFormat(CultureInfo.InvariantCulture, "\ne = {0:e}\n\n", eBest)
                .Append(les.ToString())
                .AppendFormat(CultureInfo.InvariantCulture, "LLE = {0:F5}\n", lle)
                .AppendFormat(CultureInfo.InvariantCulture, "\nBias = {0:F8}\n\n", net.NeuronBias.Memory[0]);

            for (int i = 0; i < net.Params.Neurons; i++)
            {
                debug.AppendFormat("Neuron {0} :\t\t", i + 1);
            }

            debug.AppendLine();

            foreach (double memory in net.NeuronConstant.Memory)
            {
                debug.AppendFormat(CultureInfo.InvariantCulture, "{0:F8}\t\t", memory);
            }

            debug.AppendLine();

            foreach (InputNeuron neuron in net.InputLayer.Neurons)
            {
                foreach (double memory in neuron.Memory)
                {
                    debug.AppendFormat(CultureInfo.InvariantCulture, "{0:F8}\t\t", memory);
                }

                debug.AppendLine();
            }

            debug.AppendLine();

            foreach (HiddenNeuron neuron in net.HiddenLayer.Neurons)
            {
                debug.AppendFormat(CultureInfo.InvariantCulture, "{0:F8}\t\t", neuron.Memory[0]);
            }

            debug.AppendLine("\n\n< < <");

            Logger.LogInfo(debug.ToString(), true);
        }
    }
}
