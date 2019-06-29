using System;
using System.Drawing;
using System.Globalization;
using System.Xml.Linq;
using NewMind.NeuralNet.Activation;
using NeuralAnalyser.NeuralNet.Activation;
using System.Collections.Generic;

namespace NeuralAnalyser.Configuration
{
    public class Config
    {
        public const string OptionsFile = "neural_config.xml";

        private XDocument configFile = null;

        public NeuralNetParameters NeuralNet { get; protected set; }

        public List<DataFile> Files { get; protected set; }

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

            Files = new List<DataFile>();

            foreach (var file in ConfigFile.Root.Element("FilesToAnalyse").Elements("file"))
            {
                var dataFile = GetDataFile(file);
                dataFile.Output = LoadOutParams(dataFile.FileName);
                Files.Add(dataFile);
            }
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

            var epochInterval = long.Parse(xLowParams.Attribute("epochInterval").Value, 
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
                GetActivationFunction(activationFunction), eta, epochInterval, biasTerm, constantTerm,
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

        private OutputParameters LoadOutParams(string fileName)
        {
            var output = new OutputParameters(fileName);

            var xParams = ConfigFile.Root.Element("Output");

            var xWav = xParams.Element("wav");
            var xReconstructedSignal = xParams.Element("reconstructedSignal");
            var xReconstructedPoincare = xParams.Element("reconstructedPoincare");
            var xModel3D = xParams.Element("model3D");
            var xPlots = xParams.Element("plots");
            var xAnimation = xParams.Element("animation");
            var xLeInTime = xParams.Element("leInTime");

            output.SaveWav = bool.Parse(xWav.Attribute("build").Value);

            output.PredictedSignalPts = int.Parse(xReconstructedSignal.Attribute("points").Value,
                NumberStyles.Float, CultureInfo.InvariantCulture);

            output.SaveModel = bool.Parse(xModel3D.Attribute("build").Value);

            output.SaveLeInTime = bool.Parse(xLeInTime.Attribute("build").Value);

            output.PlotsSize = new Size(
                int.Parse(xPlots.Attribute("width").Value,
                NumberStyles.Integer, CultureInfo.InvariantCulture),
                int.Parse(xPlots.Attribute("height").Value,
                NumberStyles.Integer, CultureInfo.InvariantCulture));

            output.SaveAnimation = bool.Parse(xAnimation.Attribute("build").Value);
            output.AnimationSize = new Size(
                int.Parse(xAnimation.Attribute("width").Value,
                NumberStyles.Integer, CultureInfo.InvariantCulture),
                int.Parse(xAnimation.Attribute("height").Value,
                NumberStyles.Integer, CultureInfo.InvariantCulture));

            return output;
        }

        private DataFile GetDataFile(XElement xFile)
        {
            try
            {
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

                return new DataFile(fName, dataColumn, points, start, end);
            }
            catch
            {
                throw new ArgumentException("Unable to read files list");
            }
        }
    }
}
