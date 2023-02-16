using System.Drawing;
using System.IO;

namespace NeuralNetTsa.Configuration;

public sealed class OutputParameters
{
    public OutputParameters(string fileName, int column, string outDir)
    {
        FileName = Path.GetFileName(fileName);
        OutDirectory = Path.Combine(outDir, $"nn_{FileName}_col{column}");
    }

    public string FileName { get; }

    public string OutDirectory { get; }

    public bool SaveModel { get; set; } = true;

    public int ModelPts { get; set; } = 100000;

    public bool SaveWav { get; set; } = true;

    public int WavLengthSec { get; set; } = 2;

    public int PredictedSignalPts { get; set; }

    public bool SaveLeInTime { get; set; } = true;

    public Size PlotsSize { get; set; } = new Size(640, 360);

    public bool SaveAnimation { get; set; } = true;

    public Size AnimationSize { get; set; } = new Size(360, 640);

    public string LogFile => Path.Combine(OutDirectory, "log.txt");

    public string SignalPlotFile => Path.Combine(OutDirectory, FileName + "_signal.png");

    public string PoincarePlotFile => Path.Combine(OutDirectory, FileName + "_poincare.png");

    public string PredictFile => Path.Combine(OutDirectory, FileName + ".predict");

    public string ReconstructedSignalPlotFile => Path.Combine(OutDirectory, FileName + "_reconstructed_signal.png");

    public string PredictedSignalPlotFile => Path.Combine(OutDirectory, FileName + "_prediction.png");

    public string ReconstructedPoincarePlotFile => Path.Combine(OutDirectory, FileName + "_reconstructed_poincare.png");

    public string NetPlotFile => Path.Combine(OutDirectory, FileName + "_network_plot.png");

    public string LeInTimeFile => Path.Combine(OutDirectory, FileName + "_leInTime.le");

    public string ModelFile => Path.Combine(OutDirectory, FileName + "_model.3da");

    public string WavFile => Path.Combine(OutDirectory, FileName + "_sound.wav");

    public string AnimationFile => Path.Combine(OutDirectory, FileName + "_neural_anim.gif");

    public int PtsToPredict { get; set; }

    public int PtsToTrain { get; set; }
}
