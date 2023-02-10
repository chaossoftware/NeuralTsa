using NeuralNetTsa.NeuralNet;

namespace NeuralNetTsa.Routines;

internal static class Attractor
{
    internal static AttractorData Construct(ChaosNeuralNet net, long pts)
    {
        int dimensions = net.Params.Dimensions;
        int neurons = net.Params.Neurons;

        double[] xt = new double[pts];
        double[] yt = new double[pts];
        double[] zt = new double[pts];
        double[] xlast = new double[dimensions + 1];

        for (int j = 0; j <= dimensions; j++)
        {
            xlast[j] = net.xdata[net.xdata.Length - 1 - j];
        }

        for (long t = 1; t < pts; t++)
        {
            double xnew = net.NeuronBias.LongMemory[0];

            for (int i = 0; i < neurons; i++)
            {
                double arg = net.NeuronConstant.LongMemory[i];

                for (int j = 0; j < dimensions; j++)
                {
                    arg += net.InputLayer.Neurons[j].LongMemory[i] * xlast[j - 1 + 1];
                }

                xnew += net.HiddenLayer.Neurons[i].LongMemory[0] * net.Params.ActFunction.Phi(arg);
            }

            for (int j = dimensions; j > 0; j--)
            {
                xlast[j] = xlast[j - 1];
            }

            yt[t] = xnew;
            xlast[0] = xnew;
            xt[t] = xlast[1];

            zt[t] = dimensions > 2 ? xlast[2] : 1d;
        }

        return new AttractorData(xt, yt, zt);
    }
}
