using System;
using System.Collections.Generic;
using System.Globalization;
using System.Xml.Linq;
using NeuralAnalyser.NeuralNet;
using NeuralAnalyser.NeuralNet.Activation;

namespace NeuralAnalyser
{
    public class ConfigReader
    {
        public const string OptionsFile = "neural_config.xml";

        private XDocument config = null;

        private XDocument Config
        {
            get
            {
                if (config == null)
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

                return config;
            }
        }

        public NeuralNetParameters LoadNeuralNetParams()
        {
            var xParams = Config.Root.Element("NeuralNetParams");

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

            var xLowParams = Config.Root.Element("LowLevelParams");

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

            return new NeuralNetParameters(neurons, dimensions, errorExponent, trainings, ptsToPredict, 
                GetActivationFunction(activationFunction), eta, cmax, biasTerm, constantTerm,
                maxPertrubation, nudge, pruning, testingInterval);
        }

        private ActivationFunction GetActivationFunction(string functionName) {
            switch (functionName.ToLower()) {
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

        public List<DataFile> GetFiles()
        {
            var files = new List<DataFile>();

            var fileObject = Config.Root.Element("FilesToAnalyse");

            try
            {
                foreach (XElement file in fileObject.Elements())
                {
                    string fName = file.Attribute("path").Value;

                    var dataColumn = int.Parse(file.Attribute("dataColumn").Value, 
                        NumberStyles.Integer, CultureInfo.InvariantCulture);

                    files.Add(new DataFile(fName, dataColumn));
                }
            }
            catch
            {
                throw new ArgumentException("Unable to read files list");
            }

            return files;
        }
    }

    public class DataFile
    {
        public DataFile(string fileName, int dataColumn)
        {
            this.FileName = fileName;
            this.DataColumn = dataColumn;
        }

        public string FileName { get; set; }

        public int DataColumn { get; set; }
    }
}
