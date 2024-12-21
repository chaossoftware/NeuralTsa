using ChaosSoft.Core.Data;
using ChaosSoft.NumericalMethods.Transform;
using NeuralNetTsa.Configuration;
using System.Drawing;

namespace NeuralNetTsa.Visualization;

internal static class Charts
{
    internal static ScottPlot.Plot NewPlot(Size size, string title, string xLabel, string yLabel)
    {
        ScottPlot.Plot plot = new(size.Width, size.Height);
        plot.XLabel(xLabel);
        plot.YLabel(yLabel);

        if (!string.IsNullOrEmpty(title))
        {
            plot.Title(title);
        }

        return plot;
    }

    internal static void PlotSourceSignalChart(OutputParams output, OutputPaths paths, DataSeries dataSeries)
    {
        ScottPlot.Plot signalPlot = NewPlot(output.PlotsSize, "Signal", "t", "f(t)");
        signalPlot.AddSignalXY(dataSeries.XValues, dataSeries.YValues);
        signalPlot.SaveFig(paths.SignalPlotFile);
    }

    internal static void PlotDelayedCoordinatesChart(OutputParams output, string outFile, double[] data)
    {
        ScottPlot.Plot dcPlot = NewPlot(output.PlotsSize, "Delayed coordinates", "t", "t+1");

        DelayedCoordinates.GetData(data).DataPoints
            .ForEach(dp => dcPlot.AddPoint(dp.X, dp.Y, Color.SteelBlue, 1));

        dcPlot.SaveFig(outFile);
    }
}
