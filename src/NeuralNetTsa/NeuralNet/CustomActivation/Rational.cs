namespace NeuralNetTsa.NeuralNet.CustomActivation;

public sealed class Rational : ComplexActivationFunction
{
    public override string Name => "Rational";

    public Rational() : base()
    {
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
