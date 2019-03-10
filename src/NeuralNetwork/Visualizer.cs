using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;
using System.Linq;
using MathLib.DrawEngine;
using NeuralNet.Entities;

namespace NeuralNetwork
{
    public class Visualizer
    {
        private Font font = new Font(new FontFamily("Cambria Math"), 13f);
        private SolidBrush textBrush = new SolidBrush(Color.Black);

        private readonly Brush brushInactiveNeuron = Brushes.WhiteSmoke;
        private readonly Brush brushBackground = Brushes.WhiteSmoke;
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

        public Animation NeuralAnimation;

        public Visualizer(Size size)
        {
            bitmap = new Bitmap(size.Width, size.Height);
            this.neuronSize = 0;
        }

        public Bitmap DrawBrain(SciNeuralNet net)
        {
            var iteration = net.successCount * net.Params.CMax;
            var g = Graphics.FromImage(bitmap);
            var gp = new GraphicsPath();

            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.TextRenderingHint = TextRenderingHint.AntiAlias;

            g.FillRectangle(Brushes.WhiteSmoke, new Rectangle(new Point(0, 0), bitmap.Size));
            g.DrawString(string.Format("Neurons: {0} ; Dimensions: {1}\nIteration: {2:N0}", net.Params.Neurons, net.Params.Dimensions, iteration), font, textBrush, 0f, 0f);

            int inputsCount = net.InputLayer.Neurons.Length;
            int hiddenCount = net.HiddenLayer.Neurons.Length;
            int outputsCount = net.OutputLayer.Neurons.Length;

            if (this.neuronSize == 0)
            {
                CalculateSizes(inputsCount, hiddenCount, outputsCount);
            }

            var maxSynapseValue = net.Connections[0].Max(s => s.Signal);
            var minSynapseValue = net.Connections[0].Min(s => s.Signal);

            foreach (var synapse in net.Connections[0])
            {
                var sourceCenter = GetItemCenter(synapse.IndexSource, yOffset1, xCenter1);
                var destinationCenter = GetItemCenter(synapse.IndexDestination, yOffset2, xCenter2);

                DrawSynapse(g, sourceCenter, destinationCenter, GetSynapseThickness(synapse.Signal, minSynapseValue, maxSynapseValue));
            }

            maxSynapseValue = net.Connections[1].Max(s => s.Signal);
            minSynapseValue = net.Connections[1].Min(s => s.Signal);

            foreach (var synapse in net.Connections[1])
            {
                var sourceCenter = GetItemCenter(synapse.IndexSource, yOffset2, xCenter2);
                var destinationCenter = GetItemCenter(synapse.IndexDestination, yOffset3, xCenter3);

                DrawSynapse(g, sourceCenter, destinationCenter, GetSynapseThickness(synapse.Signal, minSynapseValue, maxSynapseValue));
            }

            maxSynapseValue = net.InputLayer.Neurons.Select(n => n.Inputs[0]).Max(s => s.Signal);
            minSynapseValue = net.InputLayer.Neurons.Select(n => n.Inputs[0]).Min(s => s.Signal);

            for (int i = 0; i < inputsCount; i++)
            {
                var sourceCenter = GetItemCenter(i, yOffset1, xCenter1);
                var pointStart = new PointF(0f, sourceCenter.Y);
                var thickness = GetSynapseThickness(net.InputLayer.Neurons[i].Inputs[0].Signal, minSynapseValue, maxSynapseValue);

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
            new PointF((float)xCenter, (float)(yOffset + index * yDistance + (index + 0.5) * this.neuronSize));

        /// <summary>
        /// Calculate entity size based on count of entities in list and specified imege height
        /// </summary>
        /// <param name="height">image height</param>
        /// <param name="maxLayerItemsCount">count of entities in list</param>
        /// <returns></returns>
        private void CalculateSizes(int inputsCount, int hiddenCount, int outputsCount)
        {
            int maxLayerItemsCount = Math.Max(Math.Max(inputsCount, hiddenCount), outputsCount);

            this.neuronSize = Math.Min((float)this.bitmap.Height / (maxLayerItemsCount * 2 - 1), (float)this.bitmap.Width / 7d);
            this.maxSinapseThickness = this.neuronSize / 2d;
            var xDistance = ((float)this.bitmap.Width - 3d * this.neuronSize) / 4d;
            this.yDistance = ((float)this.bitmap.Height - maxLayerItemsCount * this.neuronSize) / (maxLayerItemsCount - 1);

            this.xCenter1 = xDistance + 0.5 * this.neuronSize;
            this.xCenter2 = 2 * xDistance + 1.5 * this.neuronSize;
            this.xCenter3 = 3 * xDistance + 2.5 * this.neuronSize;

            this.yOffset1 = (maxLayerItemsCount - inputsCount) * (this.neuronSize + this.yDistance) / 2;
            this.yOffset2 = (maxLayerItemsCount - hiddenCount) * (this.neuronSize + this.yDistance) / 2;
            this.yOffset3 = (maxLayerItemsCount - outputsCount) * (this.neuronSize + this.yDistance) / 2;
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
}
