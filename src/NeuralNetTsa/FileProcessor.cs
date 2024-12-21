using ChaosSoft.Core.Data;
using ChaosSoft.Core.IO;
using NeuralNetTsa.Configuration;
using NeuralNetTsa.NeuralNet;
using NeuralNetTsa.Visualization;
using System.IO;
using System.Linq;

namespace NeuralNetTsa;

public class FileProcessor
{
    public static void ProcessFile(Config config, DataFileParams dataFile)
    {
        NeuralNetParams neuralNetParams = config.NeuralNet;
        OutputParams output = config.Output;
        OutputPaths paths = config.Output.PathsFor(dataFile);

        IDataReader reader = new PlainTextFileReader();
        SourceData data = new(reader, dataFile.FileName);

        int startPoint = dataFile.StartPoint != -1 ? dataFile.StartPoint - 1 : 0;
        int endPoint = dataFile.EndPoint != -1 ? dataFile.EndPoint - 1 : data.LinesCount - 1;

        data.SetTimeSeries(dataFile.DataColumn - 1, startPoint, endPoint, dataFile.Points, false);

        if (!Directory.Exists(paths.OutDirectory))
        {
            Directory.CreateDirectory(paths.OutDirectory);
        }

        Logger.Init(paths.LogFile);

        Charts.PlotSourceSignalChart(output, paths, data.TimeSeries);
        Charts.PlotDelayedCoordinatesChart(output, paths.DelayedCoordPlotFile, data.TimeSeries.YValues);

        int length = data.TimeSeries.YValues.Length - output.PtsToTrain;
        var xdata = data.TimeSeries.YValues.Take(length).ToArray();

        ChaosNeuralNet neuralNet = new(neuralNetParams, xdata);
        ConsoleNetVisualizer consoleVisualizer = new(neuralNet);

        Logger.LogInfo(neuralNetParams.GetInfoFull(), true);

        consoleVisualizer.PrintNetParams(neuralNetParams);

        var calculations = new Calculations(config, dataFile, data.TimeSeries.YValues);

        if (output.SaveAnimation)
        {
            neuralNet.CycleComplete += calculations.AddAnimationFrame;
        }

        neuralNet.CycleComplete += consoleVisualizer.ReportCycle;
        neuralNet.EpochComplete += calculations.PerformCalculations;

        neuralNet.Process();

        if (output.SaveAnimation)
        {
            calculations.Visualizator.NeuralAnimation.Dispose();
        }
    }
}
