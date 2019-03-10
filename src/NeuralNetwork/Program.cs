using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.Threading;
using MathLib;
using MathLib.Data;
using MathLib.DrawEngine.Charts;
using NeuralAnalyser.NeuralNet;

namespace NeuralAnalyser
{
    public class Program
    {
        static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");
            Thread.CurrentThread.CurrentUICulture = new CultureInfo("en-US");

            var reader = new ConfigReader();
            var file = reader.GetFiles()[0];
            var data = new SourceData(file.FileName);

            data.SetTimeSeries(file.DataColumn - 1, 0, data.Length - 1, 1, false);

            NeuralOutput.Init(file.FileName);

            var signal = new SignalPlot(data.TimeSeries, new Size(848, 480), 1);
            signal.Plot().Save(NeuralOutput.SignalPlotFileName, ImageFormat.Png);

            var poincare = new MapPlot(Ext.GeneratePseudoPoincareMapData(data.TimeSeries.YValues), new Size(848, 480), 1);
            poincare.Plot().Save(NeuralOutput.PoincarePlotFileName, ImageFormat.Png);

            var neuralNetParameters = reader.LoadNeuralNetParams();
            var neuralNet = new SciNeuralNet(neuralNetParameters, data.TimeSeries.YValues);

            Logger.LogInfo(neuralNetParameters.GetInfoFull(), true);

            Console.Title = "Signal: " + NeuralOutput.FileName + " | " + neuralNetParameters.ActFunction.Name;
            Console.WriteLine("\nStarting...");

            var calculations = new Calculations(neuralNetParameters);

            neuralNet.CycleComplete += calculations.LogCycle;
            neuralNet.EpochComplete += calculations.PerformCalculations;

            neuralNet.Process();

            calculations.Visualizator.NeuralAnimation.SaveAnimation(NeuralOutput.BasePath + "_neural_anim.gif");
        }
    }
}
