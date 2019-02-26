using System;
using MathLib.DrawEngine.Charts;
using MathLib.NeuralNetwork;
using System.Drawing.Imaging;
using System.Drawing;
using MathLib;
using MathLib.DrawEngine;
using System.Threading;
using System.Globalization;
using MathLib.Data;

namespace NeuralNetwork {
    class Program {
        
        static DataReader dr = new DataReader();

        static void Main(string[] args) {

            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");
            Thread.CurrentThread.CurrentUICulture = new CultureInfo("en-US");

            DataFile file = dr.GetFiles()[0];

            SourceData sd = new SourceData(file.FileName);
            sd.SetTimeSeries(file.DataColumn - 1, 0, sd.Length - 1, 1, false);

            NeuralOutput.Init(file.FileName);


            PlotObject signal = new SignalPlot(sd.TimeSeries, new Size(848, 480), 1);
            signal.Plot().Save(NeuralOutput.SignalPlotFileName, ImageFormat.Png);

            PlotObject poincare = new MapPlot(Ext.GeneratePseudoPoincareMapData(sd.TimeSeries.YValues), new Size(848, 480), 1);
            poincare.Plot().Save(NeuralOutput.PoincarePlotFileName, ImageFormat.Png);

            NeuralNetParams taskParams = dr.LoadNeuralNetParams();
            NeuralNet task = new NeuralNet(taskParams, sd.TimeSeries.YValues);

            Logger.LogInfo(taskParams.GetInfoFull(), true);

            Console.Title = "Signal: " + NeuralOutput.FileName + " | " + taskParams.ActFunction.GetName();
            Console.WriteLine("\nStarting...");

            Charts.NeuralAnimation = new Animation();

            Calculations calc = new Calculations(NeuralNet.Params);

            NeuralNet.LoggingMethod = calc.LoggingEvent;
            NeuralNet.EndCycleMethod = calc.EndCycleEvent;

            task.RunTask();

            Charts.NeuralAnimation.SaveAnimation(NeuralOutput.BasePath + "_neural_anim.gif");
        }
    }
}
