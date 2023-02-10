using AnimatedGif;
using ChaosSoft.Core;
using ChaosSoft.Core.Data;
using ChaosSoft.Core.IO;
using ChaosSoft.Core.NumericalMethods.Lyapunov;
using ChaosSoft.Core.Transform;
using NeuralNetTsa.Configuration;
using NeuralNetTsa.NeuralNet;
using NeuralNetTsa.Routines;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Linq;

namespace NeuralNetTsa;

internal sealed class Calculations
{
    private readonly OutputParameters _outParams;
    private readonly NeuralNetEquations _systemEquations;

    private readonly double[] _originalData;

    private readonly List<double[]> _trialsHistory = new List<double[]>();
    private readonly List<double> _errors = new List<double>();

    private readonly Size _squareSize;
    private readonly Size _rectangleSize;

    private Bitmap poincare = null;
    private Bitmap signalOriginal = null;
    private Bitmap signal = null;

    public Visualizer Visualizator { get; set; }

    public Calculations(NeuralNetParameters parameters, OutputParameters outParameters, double[] originalData)
    {
        _outParams = outParameters;

        _squareSize = new Size(_outParams.AnimationSize.Width / 2, _outParams.AnimationSize.Height / 2);
        _rectangleSize = new Size(_outParams.AnimationSize.Width, _outParams.AnimationSize.Height / 4);

        Size netImageSize = new Size(_outParams.AnimationSize.Width / 8, _outParams.AnimationSize.Height / 4);
        
        Visualizator = new Visualizer(netImageSize, _outParams.SaveAnimation, outParameters.AnimationFile);

        _systemEquations = new NeuralNetEquations(parameters.Dimensions, parameters.Neurons, parameters.ActFunction);

        _originalData = originalData;
    }

    public void AddAnimationFrame(ChaosNeuralNet net)
    {
        if (_outParams.SaveAnimation)
        {
            Visualizator.NeuralAnimation.AddFrame(PrepareAnimationFrame(net), quality: GifQuality.Bit4);
        }
    }

    public void PerformCalculations(ChaosNeuralNet net)
    {
        double lle = Lle.Calculate(net);
        
        Console.WriteLine($"LLE = {lle:F5}");

        double[] leSpec = LeSpec.Calculate(net, _systemEquations);

        // TODO!!!
        //if (_outParams.SaveLeInTime)
        //{
        //    DataWriter.CreateDataFile(_outParams.LeInTimeFile, leSpec.SpectrumInTime);
        //}

        AttractorData data = Attractor.Construct(net, _outParams.PredictedSignalPts);

        try
        {
            var pseudoPoincarePlot = new ScottPlot.Plot(_outParams.PlotsSize.Width, _outParams.PlotsSize.Height);
            pseudoPoincarePlot.XLabel("t");
            pseudoPoincarePlot.YLabel("t + 1");
            pseudoPoincarePlot.Title("Pseudo poincare");

            foreach (DataPoint dp in PseudoPoincareMap.GetMapDataFrom(data.Xt).DataPoints)
            {
                pseudoPoincarePlot.AddPoint(dp.X, dp.Y, Color.SteelBlue, 1);
            }

            pseudoPoincarePlot.SaveFig(_outParams.ReconstructedPoincarePlotFile);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Prediction was not succeeded, unable to build charts: " + ex);
        }

        if (_outParams.SaveModel)
        {
            Model3D.Create3daModelFile(_outParams.ModelFile, data.Xt, data.Yt, data.Zt);
        }

        if (_outParams.SaveWav)
        {
            Sound.CreateWavFile(_outParams.WavFile, data.Yt);
        }

        if (_outParams.SaveAnimation)
        {
            ScottPlot.Plot pPlot = new ScottPlot.Plot(_squareSize.Width, _squareSize.Height);
            pPlot.XLabel("t");
            pPlot.YLabel("t + 1");
            pPlot.Title("Pseudo poincare");

            try
            {
                foreach (DataPoint dp in PseudoPoincareMap.GetMapDataFrom(data.Xt).DataPoints)
                {
                    pPlot.AddPoint(dp.X, dp.Y, Color.SteelBlue, 1);
                }

                foreach (DataPoint dp in PseudoPoincareMap.GetMapDataFrom(net.xdata).DataPoints)
                {
                    pPlot.AddPoint(dp.X, dp.Y, Color.OrangeRed, 1.5f);
                }

                poincare = pPlot.Render();
            }
            catch
            {
                pPlot.Clear();

                foreach (DataPoint dp in PseudoPoincareMap.GetMapDataFrom(net.xdata).DataPoints)
                {
                    pPlot.AddPoint(dp.X, dp.Y, Color.OrangeRed, 1.5f);
                }

                poincare = pPlot.Render();
            }


            ScottPlot.Plot signalPlot = new ScottPlot.Plot(_rectangleSize.Width, _rectangleSize.Height);
            signalPlot.XLabel("t");
            signalPlot.YLabel("f(t)");

            try
            {
                signalPlot.AddSignal(data.Xt.Take(net.xdata.Length).ToArray());
                signal = signalPlot.Render();
            }
            catch
            {
                signalPlot.Clear();
                signal = signalPlot.Render();
            }
        }

        if (_outParams.PtsToPredict > 0)
        {
            SignalPrediction.Make(net, _outParams, _originalData);
        }

        DebugInfo.Write(net, leSpec, lle);
        Console.WriteLine($"LES = {string.Join(" ", leSpec.Select(le => NumFormatter.ToShort(le)))}\t\t\t");

        Visualizator.DrawBrain(net).Save(_outParams.NetPlotFile, ImageFormat.Png);

        if (_outParams.SaveAnimation)
        {
            _trialsHistory.Add(_errors.ToArray());
            _errors.Clear();
        }
    }

    private Bitmap PrepareAnimationFrame(ChaosNeuralNet net)
    {
        if (poincare == null)
        {
            var pPlot = new ScottPlot.Plot(_squareSize.Width, _squareSize.Height);
            pPlot.XLabel("t");
            pPlot.YLabel("t + 1");
            pPlot.Title("Pseudo poincare");

            foreach (DataPoint dp in PseudoPoincareMap.GetMapDataFrom(net.xdata).DataPoints)
            {
                pPlot.AddPoint(dp.X, dp.Y, Color.OrangeRed, 1.5f);
            }

            poincare = pPlot.Render();
        }

        if (signalOriginal == null)
        {
            var signalOriginalPlot = new ScottPlot.Plot(_rectangleSize.Width, _rectangleSize.Height);
            signalOriginalPlot.AddSignal(net.xdata, color: Color.OrangeRed);
            signalOriginalPlot.XLabel("t");
            signalOriginalPlot.YLabel("f(t)");
            signalOriginalPlot.Title("Signal");

            signalOriginal = signalOriginalPlot.Render();


            var signalPlot = new ScottPlot.Plot(_rectangleSize.Width, _rectangleSize.Height);
            signalPlot.XLabel("t");
            signalPlot.YLabel("f(t)");

            signal = signalPlot.Render();
        }

        double error = Math.Log10(net.OutputLayer.Neurons[0].ShortMemory[0]);
        error = FastMath.Min(error, 0);
        error = FastMath.Max(error, -10);

        if (!_errors.Any())
        {
            _errors.Add(error);
        }

        _errors.Add(error);

        Bitmap result = new Bitmap(_outParams.AnimationSize.Width, _outParams.AnimationSize.Height);
        Bitmap netImg = Visualizator.DrawBrain(net);


        var trainingsPlot = new ScottPlot.Plot(_outParams.AnimationSize.Width / 2, _outParams.AnimationSize.Height / 2);
        trainingsPlot.XLabel("cycle #");
        trainingsPlot.YLabel("Training error (Log10)");
        trainingsPlot.Title("Trainings");
        trainingsPlot.SetAxisLimits(0, 10, -10, 0);

        foreach (var tSeries in _trialsHistory)
        {
            trainingsPlot.AddSignal(tSeries, color: Color.Gray);
        }

        trainingsPlot.AddSignal(_errors.ToArray(), color: Color.SteelBlue);

        Bitmap chart = trainingsPlot.Render();

        StringFormat stringFormat = new StringFormat
        {
            Alignment = StringAlignment.Far
        };

        Font font = new Font(new FontFamily("Cambria Math"), 11f);
        SolidBrush textBrush = new SolidBrush(Color.Black);

        using (Graphics g = Graphics.FromImage(result))
        {
            g.TextRenderingHint = TextRenderingHint.AntiAlias;
            g.DrawImage(poincare, new Point(0, 0));
            g.DrawImage(chart, new Point(_outParams.AnimationSize.Width / 2, 0));
            g.DrawImage(netImg, new Point(_outParams.AnimationSize.Width - netImg.Width - 20, 30));
            g.DrawImage(signalOriginal, new Point(0, _outParams.AnimationSize.Height / 2));
            g.DrawImage(signal, new Point(0, _outParams.AnimationSize.Height / 4 * 3));

            g.DrawString(string.Format("Dimensions: {1}\nNeurons: {0}\nIteration: {2:N0}", net.Params.Neurons, net.Params.Dimensions, net.current + net.successCount * net.Params.EpochInterval), font, textBrush, _outParams.AnimationSize.Width * 2, 0f, stringFormat);
        }

        return result;
    }
}
