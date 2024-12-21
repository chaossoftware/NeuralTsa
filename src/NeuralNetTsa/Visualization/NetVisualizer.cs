using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using AnimatedGif;
using NeuralNetTsa.NeuralNet;
using NeuralNetTsa.NeuralNet.Entities;

namespace NeuralNetTsa.Visualization;

public sealed class NetVisualizer
{
    private readonly Brush brushInactiveNeuron = Brushes.WhiteSmoke;
    private readonly Brush brushBackground = Brushes.Transparent;
    private readonly Brush brushNeuronMain = Brushes.Crimson;
    private readonly Brush brushActiveNeuron = Brushes.Orange;
    private readonly Brush brushSynapse = Brushes.OrangeRed;

    private Bitmap bitmap;

    private double neuronSize;
    private double maxSinapseThickness;
    private double yDistance;

    private double xCenter1;
    private double xCenter2;
    private double xCenter3;
    private double yOffset1;
    private double yOffset2;
    private double yOffset3;

    public AnimatedGifCreator NeuralAnimation;

    public NetVisualizer(Size size, bool animate, string animationFile)
    {
        bitmap = new Bitmap(size.Width, size.Height);
        neuronSize = 0;

        if (animate)
        {
            NeuralAnimation = AnimatedGif.AnimatedGif.Create(animationFile, 16);
        }
    }

    public Bitmap DrawBrain(ChaosNeuralNet net)
    {
        var g = Graphics.FromImage(bitmap);
        var gp = new GraphicsPath();

        g.SmoothingMode = SmoothingMode.AntiAlias;

        g.FillRectangle(Brushes.Transparent, new Rectangle(new Point(0, 0), bitmap.Size));

        int inputsCount = net.InputLayer.Neurons.Length;
        int hiddenCount = net.HiddenLayer.Neurons.Length;
        int outputsCount = net.OutputLayer.Neurons.Length;

        if (neuronSize == 0)
        {
            CalculateSizes(inputsCount, hiddenCount, outputsCount);
        }

        var maxSynapseValue = net.Connections[0].Max(s => s.Signal);
        var minSynapseValue = net.Connections[0].Min(s => s.Signal);

        foreach (var synapse in net.Connections[0])
        {
            var sourceCenter = GetItemCenter(synapse.InIndex, yOffset1, xCenter1);
            var destinationCenter = GetItemCenter(synapse.OutIndex, yOffset2, xCenter2);

            DrawSynapse(g, sourceCenter, destinationCenter, GetSynapseThickness(synapse.Signal, minSynapseValue, maxSynapseValue));
        }

        maxSynapseValue = net.Connections[1].Max(s => s.Signal);
        minSynapseValue = net.Connections[1].Min(s => s.Signal);

        foreach (var synapse in net.Connections[1])
        {
            var sourceCenter = GetItemCenter(synapse.InIndex, yOffset2, xCenter2);
            var destinationCenter = GetItemCenter(synapse.OutIndex, yOffset3, xCenter3);

            DrawSynapse(g, sourceCenter, destinationCenter, GetSynapseThickness(synapse.Signal, minSynapseValue, maxSynapseValue));
        }

        maxSynapseValue = net.InputLayer.Neurons.Select(n => n.Inputs[0]).Max(s => s.Signal);
        minSynapseValue = net.InputLayer.Neurons.Select(n => n.Inputs[0]).Min(s => s.Signal);

        for (int i = 0; i < inputsCount; i++)
        {
            var sourceCenter = GetItemCenter(i, yOffset1, xCenter1);
            var pointStart = new PointF(0f, sourceCenter.Y);

            double thickness = GetSynapseThickness(net.InputLayer.Neurons[i].Inputs[0].Signal, minSynapseValue, maxSynapseValue);

            if (double.IsNaN(thickness))
            {
                thickness = maxSinapseThickness / 2;
            }

            DrawSynapse(g, pointStart, sourceCenter, thickness);
            DrawNeuron(g, gp, net.InputLayer.Neurons[i], sourceCenter);
        }

        for (int i = 0; i < hiddenCount; i++)
        {
            var sourceCenter = GetItemCenter(i, yOffset2, xCenter2);

            DrawNeuron(g, gp, net.HiddenLayer.Neurons[i], sourceCenter);
        }

        //maxSynapseValue = net.OutputLayer.Neurons.Select(n => n.Outputs[0]).Max(s => s.Signal);
        //minSynapseValue = net.OutputLayer.Neurons.Select(n => n.Outputs[0]).Min(s => s.Signal);

        for (int i = 0; i < outputsCount; i++)
        {
            var sourceCenter = GetItemCenter(i, yOffset3, xCenter3);
            var pointStart = new PointF(bitmap.Width, sourceCenter.Y);
            var thickness = GetSynapseThickness(net.OutputLayer.Neurons[i].Outputs[0].Signal, minSynapseValue, maxSynapseValue);
            thickness = Math.Min(thickness, neuronSize);

            DrawSynapse(g, pointStart, sourceCenter, thickness);
            DrawNeuron(g, gp, net.OutputLayer.Neurons[i], sourceCenter);
        }

        g.DrawPath(new Pen(Color.FromArgb(0, Color.Black), 0), gp);
        gp.Dispose();
        g.Dispose();

        return bitmap;
    }

    private void DrawNeuron(Graphics g, GraphicsPath gp, InputNeuron neuron, PointF center)
    {
        var rect = new RectangleF((float)(center.X - neuronSize / 2), (float)(center.Y - neuronSize / 2), (float)neuronSize, (float)neuronSize);
        g.FillRectangle(GetNeuronColor(neuron), rect);
        gp.AddRectangle(rect);

        var rect1 = new RectangleF((float)(center.X - neuronSize / 2) + 3, (float)(center.Y - neuronSize / 2) + 3, (float)neuronSize - 6, (float)neuronSize - 6);
        g.FillRectangle(brushNeuronMain, rect1);
        gp.AddRectangle(rect1);
    }

    private void DrawNeuron(Graphics g, GraphicsPath gp, HiddenNeuron neuron, PointF center)
    {
        var rect = new RectangleF((float)(center.X - neuronSize / 2), (float)(center.Y - neuronSize / 2), (float)neuronSize, (float)neuronSize);
        g.FillEllipse(GetNeuronColor(neuron), rect);
        gp.AddEllipse(rect);

        var rect1 = new RectangleF((float)(center.X - neuronSize / 2) + 3, (float)(center.Y - neuronSize / 2) + 3, (float)neuronSize - 6, (float)neuronSize - 6);
        g.FillEllipse(brushNeuronMain, rect1);
        gp.AddEllipse(rect1);
    }

    private void DrawNeuron(Graphics g, GraphicsPath gp, OutputNeuron neuron, PointF center)
    {
        var rect = new RectangleF((float)(center.X - neuronSize / 2), (float)(center.Y - neuronSize / 2), (float)neuronSize, (float)neuronSize);
        g.FillEllipse(GetNeuronColor(neuron), rect);
        gp.AddEllipse(rect);

        var rect1 = new RectangleF((float)(center.X - neuronSize / 2) + 3, (float)(center.Y - neuronSize / 2) + 3, (float)neuronSize - 6, (float)neuronSize - 6);
        g.FillEllipse(brushNeuronMain, rect1);
        gp.AddEllipse(rect1);
    }

    private void DrawSynapse(Graphics g, PointF start, PointF end, double thickness)
    {
        g.DrawLine(new Pen(brushSynapse, (float)thickness), start, end);
    }

    /// <summary>
    /// Get Entity center coordinates
    /// </summary>
    /// <param name="index">index of entity in list</param>
    /// <param name="items">count of entities in current list</param>
    /// <param name="maxItems">count of entities in the longest list</param>
    /// <param name="itemSize">calculated single item size</param>
    /// <param name="xOffset">offset by X coordinate (based on entity type)</param>
    /// <returns></returns>
    private PointF GetItemCenter(int index, double yOffset, double xCenter) =>
        new PointF((float)xCenter, (float)(yOffset + index * yDistance + (index + 0.5) * neuronSize));

    /// <summary>
    /// Calculate entity size based on count of entities in list and specified imege height
    /// </summary>
    /// <param name="height">image height</param>
    /// <param name="maxLayerItemsCount">count of entities in list</param>
    /// <returns></returns>
    private void CalculateSizes(int inputsCount, int hiddenCount, int outputsCount)
    {
        int maxLayerItemsCount = Math.Max(Math.Max(inputsCount, hiddenCount), outputsCount);

        neuronSize = Math.Min((float)bitmap.Height / (maxLayerItemsCount * 2 - 1), bitmap.Width / 7d);
        maxSinapseThickness = neuronSize / 2d;
        var xDistance = (bitmap.Width - 3d * neuronSize) / 4d;
        yDistance = (bitmap.Height - maxLayerItemsCount * neuronSize) / (maxLayerItemsCount - 1);

        xCenter1 = xDistance + 0.5 * neuronSize;
        xCenter2 = 2 * xDistance + 1.5 * neuronSize;
        xCenter3 = 3 * xDistance + 2.5 * neuronSize;

        yOffset1 = (maxLayerItemsCount - inputsCount) * (neuronSize + yDistance) / 2;
        yOffset2 = (maxLayerItemsCount - hiddenCount) * (neuronSize + yDistance) / 2;
        yOffset3 = (maxLayerItemsCount - outputsCount) * (neuronSize + yDistance) / 2;
    }

    /// <summary>
    /// Get neuron color based on sign of output signal
    /// </summary>
    /// <param name="neuron">current neuron instance</param>
    /// <returns>colored brush</returns>
    private Brush GetNeuronColor(InputNeuron neuron) =>
        neuron.Outputs[0].Signal > 0 ? brushActiveNeuron : brushInactiveNeuron;

    private Brush GetNeuronColor(HiddenNeuron neuron) =>
        neuron.Outputs[0].Signal > 0 ? brushActiveNeuron : brushInactiveNeuron;

    private Brush GetNeuronColor(OutputNeuron neuron) =>
        neuron.Outputs[0].Signal > 0 ? brushActiveNeuron : brushInactiveNeuron;

    private double GetSynapseThickness(double current, double minValue, double maxValue) =>
        (current - minValue) * (maxSinapseThickness / (maxValue - minValue)) + 0.2;
}
