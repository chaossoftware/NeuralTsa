namespace NeuralNetTsa.NeuralNet.CustomActivation;

public sealed class PolynomialSixOrder : ComplexActivationFunction
{
    public override string Name => "Polynomial (6 order)";

    public PolynomialSixOrder() : base()
    {
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
