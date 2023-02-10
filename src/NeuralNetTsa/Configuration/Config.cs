using System;
using System.Drawing;
using System.Globalization;
using System.Xml.Linq;
using NeuralNetTsa.NeuralNet.Activation;
using System.Collections.Generic;
using SciML.NeuralNetwork.Activation;
using NeuralNetTsa.NeuralNet;
using System.IO;

namespace NeuralNetTsa.Configuration;

public sealed class Config
{
    private const string OptionsFile = "app_config.xml";

    private readonly XDocument _xConfig;
    private readonly CultureInfo _culture;

    public Config()
    {
        _culture = CultureInfo.InvariantCulture;

        try
        {
            _xConfig = XDocument.Load(OptionsFile);
        }
        catch
        {
            throw new ArgumentException($"Unable to load configuration file {OptionsFile}.");
        }

        LoadNeuralNetParams();

        Files = new List<DataFile>();

        foreach (var file in _xConfig.Root.Element("FilesToAnalyse").Elements("File"))
        {
            var dataFile = GetDataFile(file);
            dataFile.Output = LoadOutParams(dataFile.FileName, dataFile.DataColumn);
            Files.Add(dataFile);
        }
    }

    public NeuralNetParameters NeuralNet { get; private set; }

    public List<DataFile> Files { get; }

    private void LoadNeuralNetParams()
    {
        var xParams = _xConfig.Root.Element("NeuralNetParams");

        var neurons = ParseInt(xParams.Attribute("neurons").Value);
        var dimensions = ParseInt(xParams.Attribute("dimensions").Value);
        var activationFunction = xParams.Attribute("activationFunction").Value;
        var testingInterval = ParseLong(xParams.Attribute("testingInterval").Value);
        var epochInterval = ParseLong(xParams.Attribute("epochInterval").Value);
        var trainings = ParseInt(xParams.Attribute("trainingsCount").Value);

        var xLowParams = _xConfig.Root.Element("LowLevelParams");

        var errorExponent = ParseInt(xLowParams.Attribute("errorExponent").Value);
        var eta = ParseDouble(xLowParams.Attribute("learningRate").Value);
        var biasTerm = ParseInt(xLowParams.Attribute("biasTerm").Value);
        var constantTerm = ParseInt(xLowParams.Attribute("constantTerm").Value);
        var maxPertrubation = ParseDouble(xLowParams.Attribute("maxPertrubation").Value);
        var nudge = ParseDouble(xLowParams.Attribute("nudge").Value);
        var pruning = ParseInt(xLowParams.Attribute("pruning").Value);

        IActivationFunction activation = GetActivationFunction(activationFunction);

        NeuralNet = new NeuralNetParameters(neurons, dimensions, errorExponent, trainings,
            activation, eta, epochInterval, biasTerm, constantTerm,
            maxPertrubation, nudge, pruning, testingInterval);
    }

    private IActivationFunction GetActivationFunction(string functionName)
    {
        switch (functionName.ToLower())
        {
            case "binary_shift":
                return new BinaryShift();
            case "gaussian":
                return new Gaussian();
            case "gaussian_derivative":
                return new GaussianDerivative();
            case "sigmoid":
                return new Sigmoid();
            case "exponential":
                return new Exponential();
            case "linear":
                return new Linear();
            case "piecewise_linear":
                return new PiecewiseLinear();
            case "hyperbolic_tangent":
                return new HyperbolicTangent();
            case "hyperbolic_tangent_v2":
                return new HyperbolicTangentV2();
            case "cosine":
                return new Cosine();
            case "logistic":
                return new Logistic();
            case "logistic_v2":
                return new LogisticV2();
            case "polynomial":
                return new PolynomialSixOrder();
            case "rational":
                return new Rational();
            case "special":
                return new Special();
            case "sinc":
                return new Sinc();
            default:
                return new LogisticV2();
        }
    }

    private OutputParameters LoadOutParams(string fileName, int column)
    {
        var xParams = _xConfig.Root.Element("Output");

        string outDir = xParams.Element("Folder").Value;

        if (!Directory.Exists(outDir))
        {
            Directory.CreateDirectory(outDir);
        }

        var output = new OutputParameters(fileName, column, outDir);

        var xPrediction = xParams.Element("prediction");
        output.PtsToPredict = ParseInt(xPrediction.Attribute("predict").Value);
        output.PtsToTrain = ParseInt(xPrediction.Attribute("train").Value);

        var xReconstruction = xParams.Element("reconstruction");
        output.PredictedSignalPts = ParseFloatAsInt(xReconstruction.Attribute("points").Value);
        output.SaveWav = bool.Parse(xReconstruction.Attribute("wav").Value);
        output.SaveModel = bool.Parse(xReconstruction.Attribute("model3D").Value);

        var xPlots = xParams.Element("plots");
        var xAnimation = xParams.Element("animation");
        var xLeInTime = xParams.Element("leInTime");

        output.SaveLeInTime = bool.Parse(xLeInTime.Attribute("build").Value);

        output.PlotsSize = new Size(
            ParseInt(xPlots.Attribute("width").Value),
            ParseInt(xPlots.Attribute("height").Value));

        output.SaveAnimation = bool.Parse(xAnimation.Attribute("build").Value);
        output.AnimationSize = new Size(
            ParseInt(xAnimation.Attribute("width").Value),
            ParseInt(xAnimation.Attribute("height").Value));

        return output;
    }

    private DataFile GetDataFile(XElement xFile)
    {
        try
        {
            string fName = xFile.Attribute("path").Value;
            var dataColumn = ParseInt(xFile.Attribute("dataColumn").Value);
            var points = ParseInt(xFile.Attribute("points").Value);
            int start = TryParseInt(xFile.Attribute("start").Value);
            int end = TryParseInt(xFile.Attribute("end").Value);

            return new DataFile(fName, dataColumn, points, start, end);
        }
        catch
        {
            throw new ArgumentException("Unable to read files list");
        }
    }


    private int ParseInt(string value) =>
        int.Parse(value, NumberStyles.Integer, _culture);

    private int ParseFloatAsInt(string value) =>
        int.Parse(value, NumberStyles.Float, _culture);

    private long ParseLong(string value) =>
        long.Parse(value, NumberStyles.Float, _culture);

    private double ParseDouble(string value) =>
        double.Parse(value, NumberStyles.Float, _culture);

    private int TryParseInt(string value) =>
        int.TryParse(value, NumberStyles.Integer, _culture, out int parsed) ? 
        parsed : 
        -1;
}
