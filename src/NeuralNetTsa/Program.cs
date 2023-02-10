using ChaosSoft.Core.Data;
using ChaosSoft.Core.Transform;
using NeuralNetTsa.Configuration;
using NeuralNetTsa.NeuralNet;
using System;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;

namespace NeuralNetTsa;

internal class Program
{
    private ConsoleNetVisualizer consoleVisualizer;

    static void Main(string[] args)
    {
        Console.Title = "Neural Net Time Series Analyzer";
        Console.OutputEncoding = System.Text.Encoding.Unicode;
        Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");
        Thread.CurrentThread.CurrentUICulture = new CultureInfo("en-US");

        Program program = new Program();
        Config config = new Config();
        FileVersionInfo versionInfo = FileVersionInfo.GetVersionInfo(Assembly.GetExecutingAssembly().Location);

        foreach (var dataFile in config.Files)
        {
            Console.Clear();
            Console.WriteLine($"Version: {versionInfo.ProductVersion}");
            Console.WriteLine($"File: {dataFile.FileName}");
            program.ProcessFile(config.NeuralNet, dataFile);
        }
    }

    private void ProcessFile(NeuralNetParameters neuralNetParams, DataFile dataFile)
    {
        SourceData data = new SourceData(dataFile.FileName);

        int startPoint = dataFile.StartPoint != -1 ? dataFile.StartPoint - 1 : 0;
        int endPoint = dataFile.EndPoint != -1 ? dataFile.EndPoint - 1 : data.LinesCount - 1;

        data.SetTimeSeries(dataFile.DataColumn - 1, startPoint, endPoint, dataFile.Points, false);

        if (!Directory.Exists(dataFile.Output.OutDirectory))
        {
            Directory.CreateDirectory(dataFile.Output.OutDirectory);
        }

        Logger.Init(dataFile.Output.LogFile);

        var signalPlot = new ScottPlot.Plot(dataFile.Output.PlotsSize.Width, dataFile.Output.PlotsSize.Height);
        signalPlot.AddSignalXY(data.TimeSeries.XValues, data.TimeSeries.YValues);
        signalPlot.XLabel("t");
        signalPlot.YLabel("f(t)");
        signalPlot.Title("Signal");
        signalPlot.SaveFig(dataFile.Output.SignalPlotFile);

        var pseudoPoincarePlot = new ScottPlot.Plot(dataFile.Output.PlotsSize.Width, dataFile.Output.PlotsSize.Height);
        pseudoPoincarePlot.XLabel("t");
        pseudoPoincarePlot.YLabel("t + 1");
        pseudoPoincarePlot.Title("Pseudo poincare");

        foreach (DataPoint dp in PseudoPoincareMap.GetMapDataFrom(data.TimeSeries.YValues).DataPoints)
        {
            pseudoPoincarePlot.AddPoint(dp.X, dp.Y, Color.SteelBlue, 2);
        }

        pseudoPoincarePlot.SaveFig(dataFile.Output.PoincarePlotFile);

        int length = data.TimeSeries.YValues.Length - dataFile.Output.PtsToTrain;
        var xdata = data.TimeSeries.YValues.Take(length).ToArray();

        ChaosNeuralNet neuralNet = new ChaosNeuralNet(neuralNetParams, xdata);
        consoleVisualizer = new ConsoleNetVisualizer(neuralNet);

        Logger.LogInfo(neuralNetParams.GetInfoFull(), true);

        consoleVisualizer.PrintNetParams(neuralNetParams);

        var calculations = new Calculations(neuralNetParams, dataFile.Output, data.TimeSeries.YValues);


        if (dataFile.Output.SaveAnimation)
        {
            neuralNet.CycleComplete += calculations.AddAnimationFrame;
        }

        neuralNet.CycleComplete += consoleVisualizer.ReportCycle;
        neuralNet.EpochComplete += calculations.PerformCalculations;

        neuralNet.Process();

        if (dataFile.Output.SaveAnimation)
        {
            calculations.Visualizator.NeuralAnimation.Dispose();
        }
    }
}
