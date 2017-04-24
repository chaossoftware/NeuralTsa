using System;
using System.Collections.Generic;
using System.Xml.Linq;
using System.Globalization;
using MathLib.NeuralNetwork;

namespace NeuralNetwork
{

    public class DataReader {

        public string p_OptionsFile {
            get {
                return "neural_config.xml";
            }
        }
        
        public XDocument p_doc {
            get {
                try {
                    return XDocument.Load(p_OptionsFile);
                }
                catch {
                    throw new ArgumentException("Unable to load configuration file.");
                }
            }
        }


        public NeuralNetParams LoadNeuralNetParams() {
            int neurons, dimensions, errorExponent, trainings, ptsToPredict;
            string activationFunction;

            double eta, maxPertrubation, nudge, testingInterval;
            int biasTerm, constantTerm, pruning;
            long cmax;

            XElement nnParamsObject = p_doc.Root.Element("NeuralNetParams");
            Int32.TryParse(nnParamsObject.Attribute("neurons").Value, NumberStyles.Integer, 
                CultureInfo.InvariantCulture, out neurons);
            Int32.TryParse(nnParamsObject.Attribute("dimensions").Value, NumberStyles.Integer, 
                CultureInfo.InvariantCulture, out dimensions);
            activationFunction = nnParamsObject.Attribute("activationFunction").Value;
            Int32.TryParse(nnParamsObject.Attribute("errorExponent").Value, NumberStyles.Integer, 
                CultureInfo.InvariantCulture, out errorExponent);
            Int32.TryParse(nnParamsObject.Attribute("trainingsCount").Value, NumberStyles.Integer, 
                CultureInfo.InvariantCulture, out trainings);
            Int32.TryParse(nnParamsObject.Attribute("pointsToPredict").Value, NumberStyles.Integer, 
                CultureInfo.InvariantCulture, out ptsToPredict);

            XElement llParamsObject = p_doc.Root.Element("LowLevelParams");
            Double.TryParse(llParamsObject.Attribute("learningRate").Value, NumberStyles.Float,
                CultureInfo.InvariantCulture, out eta);
            Int64.TryParse(llParamsObject.Attribute("cycleSize").Value, NumberStyles.Float,
                CultureInfo.InvariantCulture, out cmax);
            Int32.TryParse(llParamsObject.Attribute("biasTerm").Value, NumberStyles.Integer,
                CultureInfo.InvariantCulture, out biasTerm);
            Int32.TryParse(llParamsObject.Attribute("constantTerm").Value, NumberStyles.Integer,
                CultureInfo.InvariantCulture, out constantTerm);
            Double.TryParse(llParamsObject.Attribute("maxPertrubation").Value, NumberStyles.Float,
                CultureInfo.InvariantCulture, out maxPertrubation);
            Double.TryParse(llParamsObject.Attribute("nudge").Value, NumberStyles.Float,
                CultureInfo.InvariantCulture, out nudge);
            Int32.TryParse(llParamsObject.Attribute("pruning").Value, NumberStyles.Integer,
                CultureInfo.InvariantCulture, out pruning);
            Double.TryParse(llParamsObject.Attribute("testingInterval").Value, NumberStyles.Float,
                CultureInfo.InvariantCulture, out testingInterval);

            return new NeuralNetParams(neurons, dimensions, errorExponent, trainings, ptsToPredict, 
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

        public List<DataFile> GetFiles() {
            List<DataFile> files = new List<DataFile>();

            XElement fileObject = p_doc.Root.Element("FilesToAnalyse");

            try {
                foreach (XElement file in fileObject.Elements()) {
                    string fName = file.Attribute("path").Value;
                    int dataColumn;
                    Int32.TryParse(file.Attribute("dataColumn").Value, NumberStyles.Integer, CultureInfo.InvariantCulture, out dataColumn);
                    files.Add(new DataFile(fName, dataColumn));
                }
            }
            catch {
                throw new ArgumentException("Unable to read files list");
            }

            return files;
        }

    }

    public class DataFile {
        public string FileName;
        public int DataColumn;

        public DataFile(string fileName, int dataColumn) {
            this.FileName = fileName;
            this.DataColumn = dataColumn;
        }
    }
}
