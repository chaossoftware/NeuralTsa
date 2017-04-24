using System;
using MathLib.DrawEngine.Charts;
using MathLib.NeuralNetwork;
using System.Drawing.Imaging;
using System.Drawing;
using MathLib.IO;
using MathLib;
using MathLib.DrawEngine;
using System.Threading;
using System.Globalization;

namespace NeuralNetwork {
    class Program {
        
        static DataReader dr = new DataReader();

        static void Main(string[] args) {

            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");
            Thread.CurrentThread.CurrentUICulture = new CultureInfo("en-US");

            DataFile file = dr.GetFiles()[0];

            SourceData sd = MathLib.IO.DataReader.readTimeSeries(file.FileName);
            sd.SetTimeSeries(file.DataColumn - 1, 0, sd.timeSeriesLength - 1, 1, false);

            NeuralOutput.Init(file.FileName);


            PlotObject signal = new SignalPlot(sd.TimeSeries, new Size(848, 480), 1);
            signal.Plot().Save(NeuralOutput.SignalPlotFileName, ImageFormat.Png);

            PlotObject poincare = new MapPlot(Ext.GeneratePseudoPoincareMapData(sd.TimeSeries.ValY), new Size(848, 480), 1);
            poincare.Plot().Save(NeuralOutput.PoincarePlotFileName, ImageFormat.Png);

            NeuralNetParams taskParams = dr.LoadNeuralNetParams();
            NeuralNet task = new NeuralNet(taskParams, sd.TimeSeries.ValY);

            Logger.LogInfo(taskParams.GetInfoFull(), true);

            Console.Title = "Signal: " + NeuralOutput.FileName + " | " + taskParams.ActFunction.GetName();
            Console.WriteLine("\nStarting...");

            Charts.InitAnimation();

            Calculations calc = new Calculations(NeuralNet.Task_Params);

            NeuralNet.LoggingMethod = calc.LoggingEvent;
            NeuralNet.EndCycleMethod = calc.EndCycleEvent;

            

            task.RunTask();

            Charts.SaveAnimation();
        }
    }
}
