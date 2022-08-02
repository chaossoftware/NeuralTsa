using ChaosSoft.Core.Data;
using ChaosSoft.Core.Transform;
using NeuralNetTsa.Configuration;
using NeuralNetTsa.NeuralNet;
using System;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Threading;

namespace NeuralNetTsa
{
    internal class Program
    {
        private ConsoleNetVisualizer consoleVisualizer;
        private readonly string _delimiter = new string('-', 50);

        static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");
            Thread.CurrentThread.CurrentUICulture = new CultureInfo("en-US");

            Program program = new Program();
            Config config = new Config();

            foreach (var dataFile in config.Files)
            {
                program.ProcessFile(config.NeuralNet, dataFile);
            }
        }

        private void ProcessFile(NeuralNetParameters neuralNetParams, DataFile dataFile)
        {
            Console.Title = $"{dataFile.Output.FileName} > {neuralNetParams.ActFunction.Name}";

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

            foreach (ChaosSoft.Core.Data.DataPoint dp in PseudoPoincareMap.GetMapDataFrom(data.TimeSeries.YValues).DataPoints)
            {
                pseudoPoincarePlot.AddPoint(dp.X, dp.Y, Color.SteelBlue, 2);
            }

            pseudoPoincarePlot.SaveFig(dataFile.Output.PoincarePlotFile);

            SciNeuralNet neuralNet = new SciNeuralNet(neuralNetParams, data.TimeSeries.YValues);

            Logger.LogInfo(neuralNetParams.GetInfoFull(), true);

            Console.WriteLine(neuralNetParams.GetInfoFull());
            Console.WriteLine(_delimiter);

            var calculations = new Calculations(neuralNetParams, dataFile.Output);

            consoleVisualizer = new ConsoleNetVisualizer(neuralNet);

            if (dataFile.Output.SaveAnimation)
            {
                neuralNet.CycleComplete += calculations.AddAnimationFrame;
            }

            neuralNet.CycleComplete += ReportCycle;
            neuralNet.EpochComplete += calculations.PerformCalculations;

            neuralNet.Process();

            if (dataFile.Output.SaveAnimation)
            {
                calculations.Visualizator.NeuralAnimation.SaveAnimation(dataFile.Output.AnimationFile);
            }
        }

        public void ReportCycle(SciNeuralNet net)
        {
            const int reportOffset = 19;
            double error = net.OutputLayer.Neurons[0].Memory[0];
            string currentIteration = net.current.ToString().PadRight(10);

            Console.SetCursorPosition(0, reportOffset);
            Console.WriteLine($"{currentIteration} e = {error:e}");
            Console.WriteLine(_delimiter);
            int offset = consoleVisualizer.Visualize(reportOffset + 1);
            Console.SetCursorPosition(0, offset);
            Console.WriteLine(_delimiter);
        }
    }
}
