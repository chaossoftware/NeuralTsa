using System.IO;

namespace NeuralNetTsa.Configuration;

public sealed class OutputPaths
{
    public OutputPaths(string outDir, DataFileParams dataFileParams)
    {
        FileName = Path.GetFileName(dataFileParams.FileName);
        OutDirectory = Path.Combine(outDir, $"nn_{FileName}_col-{dataFileParams.DataColumn}");
    }

    public string FileName { get; }

    public string OutDirectory { get; }

    public string LogFile => Path.Combine(OutDirectory, "log.txt");

    public string SignalPlotFile => Path.Combine(OutDirectory, FileName + "_signal.png");

    public string DelayedCoordPlotFile => Path.Combine(OutDirectory, FileName + "_delayed-coordinates.png");

    public string PredictFile => Path.Combine(OutDirectory, FileName + ".predict");

    public string ReconstSignalPlotFile => Path.Combine(OutDirectory, FileName + "_reconstructed_signal.png");

    public string PredictedSignalPlotFile => Path.Combine(OutDirectory, FileName + "_prediction.png");

    public string ReconstrDelayedCoordPlotFile => Path.Combine(OutDirectory, FileName + "_reconstructed_delayed-coordinates.png");

    public string NetPlotFile => Path.Combine(OutDirectory, FileName + "_network_plot.png");

    public string LeInTimeFile => Path.Combine(OutDirectory, FileName + "_leInTime.le");

    public string ModelFile => Path.Combine(OutDirectory, FileName + "_model.3da");

    public string WavFile => Path.Combine(OutDirectory, FileName + "_sound.wav");

    public string AnimationFile => Path.Combine(OutDirectory, FileName + "_neural_anim.gif");

    public string OverviewFile => Path.Combine(OutDirectory, FileName + "_overview.png");
}
