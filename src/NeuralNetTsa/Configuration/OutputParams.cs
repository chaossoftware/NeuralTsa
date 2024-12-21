using System.Drawing;

namespace NeuralNetTsa.Configuration;

public sealed class OutputParams
{
    public OutputParams(string outDir)
    {
        ResultsDir = outDir;
    }

    public string ResultsDir { get; }

    public bool SaveModel { get; set; } = true;

    public int ModelPts { get; set; } = 100000;

    public bool SaveWav { get; set; } = true;

    public int WavLengthSec { get; set; } = 2;

    public int PredictedSignalPts { get; set; }

    public bool SaveLeInTime { get; set; } = true;

    public Size PlotsSize { get; set; } = new Size(640, 360);

    public bool SaveAnimation { get; set; } = true;

    public Size AnimationSize { get; set; } = new Size(360, 640);

    public int PtsToPredict { get; set; }

    public int PtsToTrain { get; set; }

    public OutputPaths PathsFor(DataFileParams dataFileParams) => 
        new(ResultsDir, dataFileParams);
}
