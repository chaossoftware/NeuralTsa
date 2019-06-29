using System;
using System.Drawing;
using System.Globalization;
using System.Xml.Linq;
using NewMind.NeuralNet.Activation;
using NeuralAnalyser.NeuralNet.Activation;

namespace NeuralAnalyser.Configuration
{
    public class Config
    {
        public const string OptionsFile = "neural_config.xml";

        private XDocument configFile = null;

        public NeuralNetParameters NeuralNet { get; protected set; }

        public OutputParameters Output { get; protected set; }

        public DataFile File { get; protected set; }

        private XDocument ConfigFile
        {
            get
            {
                if (configFile == null)
                {
                    try
                    {
                        return XDocument.Load(OptionsFile);
                    }
                    catch
                    {
                        throw new ArgumentException($"Unable to load configuration file {OptionsFile}.");
                    }
                }

                return configFile;
            }
        }

        public Config()
        {
            LoadNeuralNetParams();
            LoadFile();
            LoadOutParams();
        }

        private void LoadNeuralNetParams()
        {
            var xParams = ConfigFile.Root.Element("NeuralNetParams");

            var neurons = int.Parse(xParams.Attribute("neurons").Value, 
                NumberStyles.Integer, CultureInfo.InvariantCulture);

            var dimensions = int.Parse(xParams.Attribute("dimensions").Value, 
                NumberStyles.Integer, CultureInfo.InvariantCulture);

            var activationFunction = xParams.Attribute("activationFunction").Value;

            var errorExponent = int.Parse(xParams.Attribute("errorExponent").Value, 
                NumberStyles.Integer, CultureInfo.InvariantCulture);

            var trainings = int.Parse(xParams.Attribute("trainingsCount").Value, 
                NumberStyles.Integer, CultureInfo.InvariantCulture);

            var ptsToPredict = int.Parse(xParams.Attribute("pointsToPredict").Value, 
                NumberStyles.Integer, CultureInfo.InvariantCulture);

            var xLowParams = ConfigFile.Root.Element("LowLevelParams");

            var eta = double.Parse(xLowParams.Attribute("learningRate").Value, 
                NumberStyles.Float, CultureInfo.InvariantCulture);

            var cmax = long.Parse(xLowParams.Attribute("cycleSize").Value, 
                NumberStyles.Float, CultureInfo.InvariantCulture);

            var biasTerm = int.Parse(xLowParams.Attribute("biasTerm").Value, 
                NumberStyles.Integer, CultureInfo.InvariantCulture);

            var constantTerm = int.Parse(xLowParams.Attribute("constantTerm").Value, 
                NumberStyles.Integer, CultureInfo.InvariantCulture);

            var maxPertrubation = double.Parse(xLowParams.Attribute("maxPertrubation").Value, 
                NumberStyles.Float, CultureInfo.InvariantCulture);

            var nudge = double.Parse(xLowParams.Attribute("nudge").Value, 
                NumberStyles.Float, CultureInfo.InvariantCulture);

            var pruning = int.Parse(xLowParams.Attribute("pruning").Value, 
                NumberStyles.Integer, CultureInfo.InvariantCulture);

            var testingInterval = double.Parse(xLowParams.Attribute("testingInterval").Value, 
                NumberStyles.Float, CultureInfo.InvariantCulture);

            NeuralNet = new NeuralNetParameters(neurons, dimensions, errorExponent, trainings, ptsToPredict, 
                GetActivationFunction(activationFunction), eta, cmax, biasTerm, constantTerm,
                maxPertrubation, nudge, pruning, testingInterval);
        }

        private ActivationFunction GetActivationFunction(string functionName)
        {
            switch (functionName.ToLower())
            {
                case "binary_shift":
                    return new BinaryShiftFunction();
                case "gaussian":
                    return new GaussianFunction();
                case "gaussian_derivative":
                    return new GaussianDerivativeFunction();
                case "sigmoid":
                    return new SigmoidFunction();
                case "exponential":
                    return new ExponentialFunction();
                case "linear":
                    return new LinearFunction();
                case "piecewise_linear":
                    return new PiecewiseLinearFunction();
                case "hyperbolic_tangent":
                    return new HyperbolicTangentFunction();
                case "cosine":
                    return new CosineFunction();
                case "logistic":
                    return new LogisticFunction();
                case "polynomial":
                    return new PolynomialSixOrderFunction();
                case "rational":
                    return new RationalFunction();
                case "special":
                    return new SpecialFunction();
                default:
                    return new LogisticFunction();
            }
        }

        private void LoadOutParams()
        {
            Output = new OutputParameters(File.FileName);

            var xParams = ConfigFile.Root.Element("Output");

            var xWav = xParams.Element("wav");
            var xReconstructedSignal = xParams.Element("reconstructedSignal");
            var xReconstructedPoincare = xParams.Element("reconstructedPoincare");
            var xModel3D = xParams.Element("model3D");
            var xPlots = xParams.Element("plots");
            var xAnimation = xParams.Element("animation");
            var xLeInTime = xParams.Element("leInTime");

            Output.SaveWav = bool.Parse(xWav.Attribute("build").Value);

            Output.PredictedSignalPts = int.Parse(xReconstructedSignal.Attribute("points").Value,
                NumberStyles.Float, CultureInfo.InvariantCulture);

            Output.SaveModel = bool.Parse(xModel3D.Attribute("build").Value);

            Output.SaveLeInTime = bool.Parse(xLeInTime.Attribute("build").Value);

            Output.PlotsSize = new Size(
                int.Parse(xPlots.Attribute("width").Value,
                NumberStyles.Integer, CultureInfo.InvariantCulture),
                int.Parse(xPlots.Attribute("height").Value,
                NumberStyles.Integer, CultureInfo.InvariantCulture));

            Output.SaveAnimation = bool.Parse(xAnimation.Attribute("build").Value);
            Output.AnimationSize = new Size(
                int.Parse(xAnimation.Attribute("width").Value,
                NumberStyles.Integer, CultureInfo.InvariantCulture),
                int.Parse(xAnimation.Attribute("height").Value,
                NumberStyles.Integer, CultureInfo.InvariantCulture));
        }

        private void LoadFile()
        {
            try
            {
                var xFile = ConfigFile.Root.Element("FilesToAnalyse").Element("file");

                string fName = xFile.Attribute("path").Value;

                var dataColumn = int.Parse(xFile.Attribute("dataColumn").Value, 
                    NumberStyles.Integer, CultureInfo.InvariantCulture);

                var points = int.Parse(xFile.Attribute("points").Value,
                    NumberStyles.Integer, CultureInfo.InvariantCulture);

                int start;
                int end;

                if (!int.TryParse(xFile.Attribute("start").Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out start))
                {
                    start = -1;
                }

                if (!int.TryParse(xFile.Attribute("end").Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out end))
                {
                    end = -1;
                }

                File = new DataFile(fName, dataColumn, points, start, end);
            }
            catch
            {
                throw new ArgumentException("Unable to read files list");
            }
        }
    }

    public class DataFile
    {
        public DataFile(string fileName, int dataColumn, int points, int startPoint, int endPoint)
        {
            this.FileName = fileName;
            this.DataColumn = dataColumn;
            this.StartPoint = startPoint;
            this.EndPoint = endPoint;
            this.Points = points;
        }

        public string FileName { get; set; }

        public int DataColumn { get; set; }

        public int StartPoint { get; set; }

        public int EndPoint { get; set; }

        public int Points { get; set; }
    }
}
