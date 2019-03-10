using System;

namespace NeuralAnalyser.NeuralNet.Activation
{
    public class BinaryShiftFunction : ActivationFunction
    {
        public override string Name => "Binary shift";

        public override double Phi(double arg) => arg % 1d;

        public override double Dphi(double arg) => 1d;
    }

    public class GaussianFunction : ActivationFunction
    {
        public override string Name => "Gaussian";

        public override double Phi(double arg) => arg * (1d - arg);

        public override double Dphi(double arg) => 1d - 2d * arg;
    }

    public class GaussianDerivativeFunction : ActivationFunction
    {
        public override string Name => "Gaussian Derivative";

        public override double Phi(double arg) => -arg * Math.Exp(-arg * arg);

        public override double Dphi(double arg) => (2d * arg - 1d) * Math.Exp(-arg * arg);
    }

    public class LogisticFunction : ActivationFunction
    {
        public override string Name => "Logistic";

        public override double Phi(double arg) => arg * (1d - arg);

        public override double Dphi(double arg) => 1d - 2d * arg;
    }

    public class LinearFunction : ActivationFunction
    {
        public override string Name => "Linear";

        public override double Phi(double arg) => arg;

        public override double Dphi(double arg) => 2d * arg;
    }

    public class PiecewiseLinearFunction : ActivationFunction
    {
        public override string Name => "Piecewise Linear";

        public override double Phi(double arg) => 
            Math.Abs(arg) < 1d ? arg : Math.Sign(arg);

        public override double Dphi(double arg) =>
            Math.Abs(arg) < 1d ? 1 : 0;
    }

    public class ExponentialFunction : ActivationFunction
    {
        public override string Name => "Exponential";

        public override double Phi(double arg) => Math.Exp(arg);

        public override double Dphi(double arg) => Math.Exp(arg);
    }

    public class CosineFunction : ActivationFunction
    {
        public override string Name => "Cosine";

        public override double Phi(double arg) => Math.Cos(arg);

        public override double Dphi(double arg) => Math.Cos(arg);
    }

    public class SigmoidFunction : ActivationFunction
    {
        public override string Name => "Sigmoid";

        public override double Phi(double arg)
        {
            if (arg < -44d)
            {
                return 0d;
            }
            else if (arg > 44d)
            {
                return 1d;
            }
            else
            {
                return 1d / (1d + Math.Exp(-arg));
            }
        }

        public override double Dphi(double arg)
        {
            if (Math.Abs(arg) > 44d)
            {
                return 0d;
            }
            else
            {
                double argExp = Math.Exp(arg);
                double _v = (1d + argExp);
                return argExp / (_v * _v);
            }
        }
    }

    public class HyperbolicTangentFunction : ActivationFunction
    {
        public override string Name => "Hyperbolic tangent";

        public override double Phi(double arg) =>
            arg < 22d ?  
            1d - 2d / (Math.Exp(2d * arg) + 1d) : 
            Math.Sign(arg);

        public override double Dphi(double arg)
        {
            double tmp = Sech(arg);
            return tmp * tmp;
        }
    }

    public class PolynomialSixOrderFunction : ActivationFunction
    {
        public override string Name => "Polynomial (6 order)";

        public PolynomialSixOrderFunction()
        {
            AdditionalNeuron = true;
            InitNetworkLayer();
        }

        public override double Phi(double arg) => 
            Neuron.Outputs[0].Weight + 
                arg * (Neuron.Outputs[1].Weight + 
                    arg * (Neuron.Outputs[2].Weight + 
                        arg * (Neuron.Outputs[3].Weight + 
                            arg * (Neuron.Outputs[4].Weight + 
                                arg * (Neuron.Outputs[5].Weight + 
                                    arg * Neuron.Outputs[6].Weight)))));

        public override double Dphi(double arg) =>
            Neuron.Outputs[1].Weight + 
                arg * (2d * Neuron.Outputs[2].Weight + 
                    arg * (3d * Neuron.Outputs[3].Weight + 
                        arg * (4d * Neuron.Outputs[4].Weight + 
                            arg * (5d * Neuron.Outputs[5].Weight + 
                                arg * 6d * Neuron.Outputs[6].Weight))));
    }

    public class RationalFunction : ActivationFunction
    {
        public override string Name => "Rational";

        public RationalFunction()
        {
            AdditionalNeuron = true;
            InitNetworkLayer();
        }

        public override double Phi(double arg) =>
            (Neuron.Outputs[0].Weight + 
                arg * (Neuron.Outputs[1].Weight + 
                    arg * (Neuron.Outputs[2].Weight + 
                        arg * Neuron.Outputs[3].Weight))) 
                / (1d + 
                arg * (Neuron.Outputs[4].Weight + 
                    arg * (Neuron.Outputs[5].Weight + 
                        arg * Neuron.Outputs[6].Weight)));

        public override double Dphi(double arg)
        {
            double f = Neuron.Outputs[0].Weight + 
                arg * (Neuron.Outputs[1].Weight + 
                    arg * (Neuron.Outputs[2].Weight + 
                        arg * Neuron.Outputs[3].Weight));

            double df = Neuron.Outputs[1].Weight + 
                arg * (2d * Neuron.Outputs[2].Weight + 
                    arg * 3d * Neuron.Outputs[3].Weight);

            double g = 1d + arg * (Neuron.Outputs[4].Weight + 
                arg * (Neuron.Outputs[5].Weight + 
                    arg * Neuron.Outputs[6].Weight));

            double dg = Neuron.Outputs[4].Weight + 
                arg * (2d * Neuron.Outputs[5].Weight + 
                    arg * 3d * Neuron.Outputs[6].Weight);

            return (g * df - f * dg) / (g * g);
        }
    }

    public class SpecialFunction : ActivationFunction
    {
        public override string Name => "Special";

        public SpecialFunction()
        {
            AdditionalNeuron = true;
            InitNetworkLayer();
        }

        public override double Phi(double arg) =>
            Math.Abs(arg) < 22d ?
            
            Neuron.Outputs[0].Weight + 
                    arg * (Neuron.Outputs[1].Weight + 
                        arg * Neuron.Outputs[2].Weight) +
                    Neuron.Outputs[3].Weight * 
                        (1d - 2d / (Math.Exp(2d * arg) + 1d)) :
                
            Neuron.Outputs[0].Weight + 
                    arg * (Neuron.Outputs[1].Weight + 
                        arg * Neuron.Outputs[2].Weight) +
                            Neuron.Outputs[3].Weight * Math.Sign(arg);

        public override double Dphi(double arg) =>
            Neuron.Outputs[1].Weight + 
                arg * 2d * Neuron.Outputs[2].Weight +
                    Neuron.Outputs[3].Weight * Math.Pow(Sech(arg), 2);
    }
}
