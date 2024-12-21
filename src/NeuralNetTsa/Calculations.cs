using AnimatedGif;
using ChaosSoft.Core.Data;
using ChaosSoft.NumericalMethods.Transform;
using NeuralNetTsa.Configuration;
using NeuralNetTsa.NeuralNet;
using NeuralNetTsa.Routines;
using NeuralNetTsa.Visualization;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Text;
using System.Linq;

namespace NeuralNetTsa;

internal sealed class Calculations
{
    private readonly OutputParams _outParams;
    private readonly OutputPaths _paths;

    private readonly double[] _originalData;

    private readonly List<double[]> _trialsHistory = new List<double[]>();
    private readonly List<double> _errors = new List<double>();

    private readonly Size _squareSize;
    private readonly Size _rectangleSize;

    private Bitmap poincare = null;
    private Bitmap signalOriginal = null;
    private Bitmap signal = null;

    public NetVisualizer Visualizator { get; set; }

    public Calculations(Config config, DataFileParams dataFileParams, double[] originalData)
    {
        _outParams = config.Output;
        _paths = config.Output.PathsFor(dataFileParams);

        _squareSize = new Size(_outParams.AnimationSize.Width / 2, _outParams.AnimationSize.Height / 2);
        _rectangleSize = new Size(_outParams.AnimationSize.Width, _outParams.AnimationSize.Height / 4);

        Size netImageSize = new Size(_outParams.AnimationSize.Width / 8, _outParams.AnimationSize.Height / 4);
        
        Visualizator = new NetVisualizer(netImageSize, _outParams.SaveAnimation, _paths.AnimationFile);

        _originalData = originalData;
    }

    public void AddAnimationFrame(ChaosNeuralNet net)
    {
        Bitmap overview = PrepareAnimationFrame(net);
        overview.Save(_paths.OverviewFile, ImageFormat.Png);

        if (_outParams.SaveAnimation)
        {
            Visualizator.NeuralAnimation.AddFrame(overview, quality: GifQuality.Bit4);
        }
    }

    public void PerformCalculations(ChaosNeuralNet net)
    {
        double lle = Lle.Calculate(net);
        LeSpecCalculator leSpecCalculator = new();
        leSpecCalculator.Calculate(net);

        // TODO!!!
        //if (_outParams.SaveLeInTime)
        //{
        //    DataWriter.CreateDataFile(_outParams.LeInTimeFile, leSpec.SpectrumInTime);
        //}

        AttractorData data = Attractor.Construct(net, _outParams.PredictedSignalPts);

        try
        {
            Charts.PlotDelayedCoordinatesChart(_outParams, _paths.ReconstrDelayedCoordPlotFile, data.Xt);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Prediction was not succeeded, unable to build charts: " + ex);
        }

        if (_outParams.SaveModel)
        {
            Model3D.Create3daModelFile(_paths.ModelFile, data.Xt, data.Yt, data.Zt);
        }

        if (_outParams.SaveWav)
        {
            Sound.CreateWavFile(_paths.WavFile, data.Yt);
        }

        ScottPlot.Plot pPlot = Charts.NewPlot(_squareSize, "Delayed coordinates", "t", "t+1");
        pPlot.Grid(enable: false);

        try
        {
            DelayedCoordinates.GetData(data.Xt).DataPoints
                .ForEach(dp => pPlot.AddPoint(dp.X, dp.Y, Color.SteelBlue, 1));

            DelayedCoordinates.GetData(net.xdata).DataPoints
                .ForEach(dp => pPlot.AddPoint(dp.X, dp.Y, Color.OrangeRed, 1.5f));

            poincare = pPlot.Render();
        }
        catch
        {
            pPlot.Clear();

            DelayedCoordinates.GetData(net.xdata).DataPoints
                .ForEach(dp => pPlot.AddPoint(dp.X, dp.Y, Color.OrangeRed, 1.5f));

            poincare = pPlot.Render();
        }

        ScottPlot.Plot signalPlot = Charts.NewPlot(_rectangleSize, "", "t", "f(t)");

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

        if (_outParams.PtsToPredict > 0)
        {
            SignalPrediction.Make(net, _outParams, _paths, _originalData);
        }

        DebugInfo.Write(net, leSpecCalculator.Result, lle);

        Visualizator.DrawBrain(net).Save(_paths.NetPlotFile, ImageFormat.Png);

        _trialsHistory.Add(_errors.ToArray());
        _errors.Clear();
    }

    private Bitmap PrepareAnimationFrame(ChaosNeuralNet net)
    {
        if (poincare == null)
        {
            ScottPlot.Plot pPlot = Charts.NewPlot(_squareSize, "Delayed coordinates", "t", "t+1");
            pPlot.Grid(enable: false);

            foreach (DataPoint dp in DelayedCoordinates.GetData(net.xdata).DataPoints)
            {
                pPlot.AddPoint(dp.X, dp.Y, Color.OrangeRed, 1.5f);
            }

            poincare = pPlot.Render();
        }

        if (signalOriginal == null)
        {
            ScottPlot.Plot signalOriginalPlot = Charts.NewPlot(_rectangleSize, "Signal", "t", "f(t)");
            signalOriginalPlot.Grid(enable: false);

            signalOriginalPlot.AddSignal(net.xdata, color: Color.OrangeRed);

            signalOriginal = signalOriginalPlot.Render();

            ScottPlot.Plot signalPlot = Charts.NewPlot(_rectangleSize, "", "t", "f(t)");

            signal = signalPlot.Render();
        }

        double error = Math.Log10(net.OutputLayer.Neurons[0].ShortMemory[0]);
        error = Math.Min(error, 0);
        error = Math.Max(error, -10);

        if (!_errors.Any())
        {
            _errors.Add(error);
        }

        _errors.Add(error);

        Bitmap result = new(_outParams.AnimationSize.Width, _outParams.AnimationSize.Height);
        Bitmap netImg = Visualizator.DrawBrain(net);


        var trainingsPlot = new ScottPlot.Plot(_outParams.AnimationSize.Width / 2, _outParams.AnimationSize.Height / 2);
        trainingsPlot.XLabel("cycle #");
        trainingsPlot.YLabel("Training error (Log10)");
        trainingsPlot.Title("Trainings");
        trainingsPlot.SetAxisLimits(0, 10, -10, 0);
        trainingsPlot.Grid(enable: false);

        foreach (var tSeries in _trialsHistory)
        {
            var trial = trainingsPlot.AddSignal(tSeries, color: Color.Gray);
            trial.MarkerSize = 1f;
        }

        var lastTrial = trainingsPlot.AddSignal(_errors.ToArray(), color: Color.SteelBlue);
        lastTrial.MarkerSize = 4f;

        Bitmap chart = trainingsPlot.Render();

        StringFormat stringFormat = new StringFormat
        {
            Alignment = StringAlignment.Far
        };

        Font font = new(new FontFamily("Cambria Math"), 11f);
        SolidBrush textBrush = new(Color.Black);

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
