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

            foreach (var dataFile in config.Files)
            {
                ProcessFile(config.NeuralNet, dataFile);
            }
        }

        private static void ProcessFile(NeuralNetParameters neuralNetParams, DataFile dataFile)
        {
            var data = new SourceData(dataFile.FileName);

            var startPoint = dataFile.StartPoint != -1 ? dataFile.StartPoint - 1 : 0;
            var endPoint = dataFile.EndPoint != -1 ? dataFile.EndPoint - 1 : data.LinesCount - 1;

            data.SetTimeSeries(dataFile.DataColumn - 1, startPoint, endPoint, dataFile.Points, false);

            if (!Directory.Exists(dataFile.Output.OutDirectory))
            {
                Directory.CreateDirectory(dataFile.Output.OutDirectory);
            }

            Logger.Init(dataFile.Output.LogFile);

            new MathChart(dataFile.Output.PlotsSize, "t", "f(t)")
                .AddTimeSeries("Signal", data.TimeSeries, SeriesChartType.Line)
                .SaveImage(dataFile.Output.SignalPlotFile, ImageFormat.Png);

            new MathChart(dataFile.Output.PlotsSize, "f(t)", "f(t+1)")
                .AddTimeSeries("Pseudo Poincare", PseudoPoincareMap.GetMapDataFrom(data.TimeSeries.YValues), SeriesChartType.Point)
                .SaveImage(dataFile.Output.PoincarePlotFile, ImageFormat.Png);

            var neuralNet = new SciNeuralNet(neuralNetParams, data.TimeSeries.YValues);

            Logger.LogInfo(neuralNetParams.GetInfoFull(), true);

            Console.Title = "Signal: " + dataFile.Output.FileName + " | " + neuralNetParams.ActFunction.Name;
            Console.WriteLine(neuralNetParams.GetInfoFull());
            Console.WriteLine("\n\nStarting...");

            var calculations = new Calculations(neuralNetParams, dataFile.Output);

            neuralNet.CycleComplete += calculations.LogCycle;
            neuralNet.EpochComplete += calculations.PerformCalculations;

            neuralNet.Process();

            if (dataFile.Output.SaveAnimation)
            {
                calculations.Visualizator.NeuralAnimation.SaveAnimation(dataFile.Output.AnimationFile);
            }
        }
    }
}
