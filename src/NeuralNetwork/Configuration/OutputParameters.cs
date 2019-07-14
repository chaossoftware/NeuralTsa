using System.Drawing;
using System.IO;

namespace NeuralAnalyser.Configuration
{
    public class OutputParameters
    {
        public OutputParameters(string fileName)
        {
            OutDirectory = fileName + "_nnOut";
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

        public string LogFile => Path.Combine(OutDirectory, "log.txt");

        public string SignalPlotFile => Path.Combine(OutDirectory, FileName + "_signal.png");

        public string PoincarePlotFile => Path.Combine(OutDirectory, FileName + "_poincare.png");

        public string PredictFile => Path.Combine(OutDirectory, FileName + ".predict");

        public string ReconstructedSignalPlotFile => Path.Combine(OutDirectory, FileName + "_reconstructed_signal.png");

        public string PredictedSignalPlotFile => Path.Combine(OutDirectory, FileName + "_reconstructed_signal.png");

        public string ReconstructedPoincarePlotFile => Path.Combine(OutDirectory, FileName + "_reconstructed_poincare.png");

        public string NetPlotFile => Path.Combine(OutDirectory, FileName + "_network_plot.png");

        public string LeInTimeFile => Path.Combine(OutDirectory, FileName + "_leInTime.le");

        public string ModelFile => Path.Combine(OutDirectory, FileName + "_model.3da");

        public string WavFile => Path.Combine(OutDirectory, FileName + "_sound.wav");

        public string AnimationFile => Path.Combine(OutDirectory, FileName + "_neural_anim.gif");
    }
}
