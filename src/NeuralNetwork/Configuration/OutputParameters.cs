using System.Drawing;
using System.IO;

namespace NeuralAnalyser.Configuration
{
    public class OutputParameters
    {
        public OutputParameters(string fileName)
        {
            OutDirectory = fileName + "_out";
            FileName = Path.GetFileName(fileName);
        }

        public string FileName { get; protected set; }

        public string OutDirectory { get; protected set; }

        public bool SaveModel { get; set; } = true;

        public int ModelPts { get; set; } = 100000;

        public bool SaveWav { get; set; } = true;

        public int WavLengthSec { get; set; } = 2;

        public int PredictedSignalPts { get; set; }

        public bool SaveLeInTime { get; set; } = true;

        public Size PlotsSize { get; set; } = new Size(640, 360);

        public bool SaveAnimation { get; set; } = true;

        public Size AnimationSize { get; set; } = new Size(360, 640);

        public string LogFileName => Path.Combine(OutDirectory, "log.txt");

        public string SignalPlotFileName => Path.Combine(OutDirectory, FileName + "_signal.png");

        public string PoincarePlotFileName => Path.Combine(OutDirectory, FileName + "_poincare.png");

        public string PredictFileName => Path.Combine(OutDirectory, FileName + ".predict");

        public string ReconstructedSignalPlotFileName => Path.Combine(OutDirectory, FileName + "_reconstructed_signal.png");

        public string ReconstructedPoincarePlotFileName => Path.Combine(OutDirectory, FileName + "_reconstructed_poincare.png");

        public string PredictedSignalPlotFileName => Path.Combine(OutDirectory, FileName + "_reconstructed_signal.png");

        public string NetworkPlotPlotFileName => Path.Combine(OutDirectory, FileName + "_network_plot.png");

        public string LeInTimeFileName => Path.Combine(OutDirectory, FileName + "_leInTime.le");

        public string ModelFileName => Path.Combine(OutDirectory, FileName + "_model.ply");

        public string WavFileName => Path.Combine(OutDirectory, FileName + "_sound.wav");

        public string AnimationFileName => Path.Combine(OutDirectory, FileName + "_neural_anim.gif");
    }
}
