using System;
using System.Drawing;
using System.Globalization;
using System.Xml.Linq;
using System.Collections.Generic;
using ChaosSoft.NeuralNetwork.Activation;
using System.IO;
using NeuralNetTsa.NeuralNet.CustomActivation;

namespace NeuralNetTsa.Configuration;

public sealed class Config
{
    private const string OptionsFile = "app_config.xml";
    
    private static readonly CultureInfo Culture = CultureInfo.InvariantCulture;

    private readonly XDocument _xConfig;

    public Config()
    {
        try
        {
            _xConfig = XDocument.Load(OptionsFile);
        }
        catch
        {
            throw new ArgumentException($"Unable to load configuration file {OptionsFile}.");
        }
        LoadNeuralNetParams();
        LoadOutParams();

        Files = new List<DataFileParams>();

        foreach (var file in _xConfig.Root.Element("FilesToAnalyse").Elements("File"))
        {
            var dataFile = GetDataFile(file);
            Files.Add(dataFile);
        }
    }

    public NeuralNetParams NeuralNet { get; private set; }

    public List<DataFileParams> Files { get; }

    public OutputParams Output { get; set; }

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

        NeuralNet = new NeuralNetParams(neurons, dimensions, errorExponent, trainings,
            activation, eta, epochInterval, biasTerm, constantTerm,
            maxPertrubation, nudge, pruning, testingInterval);
    }

    private static IActivationFunction GetActivationFunction(string functionName)
    {
        return functionName.ToLower() switch
        {
            "binary_shift" => new BinaryShift(),
            "gaussian" => new Gaussian(),
            "gaussian_derivative" => new GaussianDerivative(),
            "sigmoid" => new Sigmoid(),
            "exponential" => new Exponential(),
            "linear" => new Linear(),
            "piecewise_linear" => new PiecewiseLinear(),
            "hyperbolic_tangent" => new HyperbolicTangent(),
            "hyperbolic_tangent_v2" => new HyperbolicTangentV2(),
            "cosine" => new Cosine(),
            "logistic" => new Logistic(),
            "logistic_v2" => new LogisticV2(),
            "polynomial" => new PolynomialSixOrder(),
            "rational" => new Rational(),
            "special" => new Special(),
            "sinc" => new Sinc(),
            _ => new LogisticV2(),
        };
    }

    private void LoadOutParams()
    {
        var xParams = _xConfig.Root.Element("Output");

        string outDir = xParams.Element("Folder").Value;

        if (!Directory.Exists(outDir))
        {
            Directory.CreateDirectory(outDir);
        }

        Output = new OutputParams(outDir);

        var xPrediction = xParams.Element("prediction");
        Output.PtsToPredict = ParseInt(xPrediction.Attribute("predict").Value);
        Output.PtsToTrain = ParseInt(xPrediction.Attribute("train").Value);

        var xReconstruction = xParams.Element("reconstruction");
        Output.PredictedSignalPts = ParseFloatAsInt(xReconstruction.Attribute("points").Value);
        Output.SaveWav = bool.Parse(xReconstruction.Attribute("wav").Value);
        Output.SaveModel = bool.Parse(xReconstruction.Attribute("model3D").Value);

        var xPlots = xParams.Element("plots");
        var xAnimation = xParams.Element("animation");
        var xLeInTime = xParams.Element("leInTime");

        Output.SaveLeInTime = bool.Parse(xLeInTime.Attribute("build").Value);

        Output.PlotsSize = new Size(
            ParseInt(xPlots.Attribute("width").Value),
            ParseInt(xPlots.Attribute("height").Value));

        Output.SaveAnimation = bool.Parse(xAnimation.Attribute("build").Value);
        Output.AnimationSize = new Size(
            ParseInt(xAnimation.Attribute("width").Value),
            ParseInt(xAnimation.Attribute("height").Value));
    }

    private static DataFileParams GetDataFile(XElement xFile)
    {
        try
        {
            string fName = xFile.Attribute("path").Value;
            var dataColumn = ParseInt(xFile.Attribute("dataColumn").Value);
            var points = ParseInt(xFile.Attribute("points").Value);
            int start = TryParseInt(xFile.Attribute("start").Value);
            int end = TryParseInt(xFile.Attribute("end").Value);

            return new DataFileParams(fName, dataColumn, points, start, end);
        }
        catch
        {
            throw new ArgumentException("Unable to read files list");
        }
    }
    
    private static int ParseInt(string value) =>
        int.Parse(value, NumberStyles.Integer, Culture);

    private static int ParseFloatAsInt(string value) =>
        int.Parse(value, NumberStyles.Float, Culture);

    private static long ParseLong(string value) =>
        long.Parse(value, NumberStyles.Float, Culture);

    private static double ParseDouble(string value) =>
        double.Parse(value, NumberStyles.Float, Culture);

    private static int TryParseInt(string value) =>
        int.TryParse(value, NumberStyles.Integer, Culture, out int parsed) ? 
        parsed : 
        -1;
}
