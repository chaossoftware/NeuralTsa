using System;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Windows.Forms.DataVisualization.Charting;
using MathLib.Data;
using MathLib.DrawEngine;
using MathLib.Transform;
using NeuralAnalyser.Configuration;
using NeuralAnalyser.NeuralNet;

namespace NeuralAnalyser
{
    public class Program
    {
        static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");
            Thread.CurrentThread.CurrentUICulture = new CultureInfo("en-US");

            var config = new Config();
            var data = new SourceData(config.File.FileName);

            var startPoint = config.File.StartPoint != -1 ? config.File.StartPoint - 1 : 0;
            var endPoint = config.File.EndPoint != -1 ? config.File.EndPoint - 1 : data.Length - 1;

            data.SetTimeSeries(config.File.DataColumn - 1, startPoint, endPoint, config.File.Points, false);

            if (!Directory.Exists(config.Output.OutDirectory))
            {
                Directory.CreateDirectory(config.Output.OutDirectory);
            }

            Logger.Init(config.Output.LogFile);

            new MathChart(config.Output.PlotsSize, "t", "f(t)")
                .AddTimeSeries("Signal", data.TimeSeries, SeriesChartType.Line)
                .SaveImage(config.Output.SignalPlotFile, ImageFormat.Png);

            new MathChart(config.Output.PlotsSize, "f(t)", "f(t+1)")
                .AddTimeSeries("Pseudo Poincare", PseudoPoincareMap.GetMapDataFrom(data.TimeSeries.YValues), SeriesChartType.Point)
                .SaveImage(config.Output.PoincarePlotFile, ImageFormat.Png);

            var neuralNetParameters = config.NeuralNet;
            var neuralNet = new SciNeuralNet(neuralNetParameters, data.TimeSeries.YValues);

            Logger.LogInfo(neuralNetParameters.GetInfoFull(), true);

            Console.Title = "Signal: " + config.Output.FileName + " | " + neuralNetParameters.ActFunction.Name;
            Console.WriteLine(neuralNetParameters.GetInfoFull());
            Console.WriteLine("\n\nStarting...");

            var calculations = new Calculations(neuralNetParameters, config.Output);

            neuralNet.CycleComplete += calculations.LogCycle;
            neuralNet.EpochComplete += calculations.PerformCalculations;

            neuralNet.Process();

            if (config.Output.SaveAnimation)
            {
                calculations.Visualizator.NeuralAnimation.SaveAnimation(config.Output.AnimationFile);
            }
        }
    }
}
