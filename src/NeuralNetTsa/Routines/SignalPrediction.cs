using NeuralNetTsa.NeuralNet;

namespace NeuralNetTsa.Routines
{
    internal static class SignalPrediction
    {
        internal static double[] Make(SciNeuralNet net, int pointsToPredict)
        {
            int dimensions = net.Params.Dimensions;
            int neurons = net.Params.Neurons;

            double[] xpred = new double[pointsToPredict + dimensions];
            double predPt;

            for (int j = 0; j < dimensions; j++)
            {
                xpred[dimensions - j] = net.xdata[net.xdata.Length - 1 - j];
            }

            for (int k = dimensions; k < pointsToPredict + dimensions; k++)
            {
                predPt = net.NeuronBias.Best[0];

                for (int i = 0; i < neurons; i++)
                {
                    double arg = net.NeuronConstant.Best[i];

                    for (int j = 0; j < dimensions; j++)
                    {
                        arg += net.InputLayer.Neurons[j].Best[i] * xpred[k - j];
                    }

                    predPt += net.HiddenLayer.Neurons[i].Best[0] * net.Params.ActFunction.Phi(arg);
                }

                xpred[k] = predPt;
            }

            return xpred;
        }
    }
}
