using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using MathLib.Data;
using MathLib.DrawEngine;
using MathLib.NumericalMethods.Lyapunov;
using MathLib.Transform;
using MathWorks.MATLAB.NET.Arrays;

namespace TimeSeriesAnalysis
{
    internal class Routines
    {
        public SourceData SourceData { get; set; }

        public LyapunovMethod Lyapunov { get; set; }

        public void FillSignalChart(MathChart chart) =>
            chart.ClearChart()
            .SetAxisNames("t", "f(t)")
            .AddTimeSeries("Signal", SourceData.TimeSeries, SeriesChartType.Line);

        public void FillPoincareChart(MathChart chart) =>
            chart.ClearChart()
            .SetAxisNames("f(t)", "f(t+1)")
            .AddTimeSeries("Pseudo Poincare Section", PseudoPoincareMap.GetMapDataFrom(SourceData.TimeSeries.YValues), SeriesChartType.Point);

        public void FillFourierChart(MathChart chart, double statrFreq, double endFreq, double dt, bool logScale)
        {
            try
            {
                var timeseries = Fourier.GetFourier(SourceData.TimeSeries.YValues, statrFreq, endFreq, dt, Convert.ToInt32(logScale));
                chart.ClearChart()
                    .SetAxisNames("ω", "f(ω)")
                    .AddTimeSeries("Fourier Spectrum", timeseries, SeriesChartType.Point);
            }
            catch (Exception ex)
            {
                chart.ClearChart();
                MessageBox.Show($"Unable to build {StringData.Fourier}:\n" + ex.Message);
            }
        }

        public string FillLyapunovChart(MathChart chart, int startPoint, int endPoint, bool isWolf)
        {
            chart.ClearChart();

            int range = endPoint - startPoint + 1;
            var result = string.Empty;

            if (isWolf)
            {
                var timeseries = new Timeseries();

                for (int i = startPoint; i < range; i++)
                {
                    timeseries.AddDataPoint(Lyapunov.Slope.DataPoints[i].X, Lyapunov.Slope.DataPoints[i].Y);
                }

                chart.SetAxisNames("t", "LE")
                    .AddTimeSeries("Lyapunov Exponent in Time", timeseries, SeriesChartType.Line);
            }
            else
            {
                var tsSector = new Timeseries();

                tsSector.AddDataPoint(Lyapunov.Slope.DataPoints[startPoint].X, Lyapunov.Slope.DataPoints[startPoint].Y);
                tsSector.AddDataPoint(Lyapunov.Slope.DataPoints[range - 1].X, Lyapunov.Slope.DataPoints[range - 1].Y);

                chart.SetAxisNames("t", "Slope")
                    .AddTimeSeries("Lyapunov Function", Lyapunov.Slope, SeriesChartType.Line)
                    .AddTimeSeries("Sector", tsSector, SeriesChartType.Line, Color.Red);

                result = string.Format("{0:F5}",
                    (Lyapunov.Slope.DataPoints[endPoint].Y - Lyapunov.Slope.DataPoints[startPoint].Y) / (Lyapunov.Slope.DataPoints[endPoint].X - Lyapunov.Slope.DataPoints[startPoint].X)
                );
            }

            return result;
        }

        public void BuildWavelet(string tmpFileName, string wName, double tStart, double tEnd, double fStart, double fEnd, double dt, string colMap, double width, double height)
        {
            var matlabEngine = new MatlabEngine.MatlabEngine();
            var mwFolder = new MWCharArray(string.Empty);
            var mwfileName = new MWCharArray(tmpFileName);
            var mwWname = new MWCharArray(wName);
            var mwColMap = new MWCharArray(colMap);
            var mwSignalArray = new MWNumericArray(SourceData.TimeSeries.YValues);

            matlabEngine.Get2DWavelet(mwSignalArray, mwFolder, mwfileName, mwWname, tStart, tEnd, fStart, fEnd, dt, mwColMap, width, height);
        }

        public void DeleteTempFiles()
        {
            File.Delete(StringData.WaveletFile);
            File.Delete(StringData.WaveletPreviewFile);
        }
    }
}
