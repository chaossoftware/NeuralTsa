using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using MathLib.Data;
using MathLib.DrawEngine;
using MathLib.MathMethods.Lyapunov;
using MathLib.Transform;
using MathWorks.MATLAB.NET.Arrays;

namespace TimeSeriesAnalysis
{
    internal class Routines
    {
        public SourceData sourceData;

        public LyapunovMethod lyapunov;

        public void FillSignalChart(MathChart chart)
        {
            chart.ClearChart();
            chart.ChartAreas[0].Axes[0].Title = "t";
            chart.ChartAreas[0].Axes[1].Title = "f(t)";
            chart.AddTimeSeries("Signal", sourceData.TimeSeries, SeriesChartType.Line);
        }

        public void FillPoincareChart(MathChart chart)
        {
            chart.ClearChart();
            chart.ChartAreas[0].Axes[0].Title = "f(t)";
            chart.ChartAreas[0].Axes[1].Title = "f(t+1)";
            chart.AddTimeSeries("Pseudo Poincare Section", PseudoPoincareMap.GetMapDataFrom(sourceData.TimeSeries.YValues), SeriesChartType.Point);
        }

        public void FillFourierChart(MathChart chart, double statrFreq, double endFreq, double dt, bool logScale)
        {
            try
            {
                var timeseries = Fourier.GetFourier(sourceData.TimeSeries.YValues, statrFreq, endFreq, dt, Convert.ToInt32(logScale));
                chart.ClearChart();
                chart.AddTimeSeries("Fourier Spectrum", timeseries, SeriesChartType.Point);
                chart.ChartAreas[0].Axes[0].Title = "ω";
                chart.ChartAreas[0].Axes[1].Title = "f(ω)";
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
                    timeseries.AddDataPoint(lyapunov.Slope.DataPoints[i].X, lyapunov.Slope.DataPoints[i].Y);
                }

                chart.AddTimeSeries("Lyapunov Exponent in Time", timeseries, SeriesChartType.Line);
                chart.ChartAreas[0].Axes[0].Title = "t";
                chart.ChartAreas[0].Axes[1].Title = "LE";
            }
            else
            {
                var tsSector = new Timeseries();

                tsSector.AddDataPoint(lyapunov.Slope.DataPoints[startPoint].X, lyapunov.Slope.DataPoints[startPoint].Y);
                tsSector.AddDataPoint(lyapunov.Slope.DataPoints[range - 1].X, lyapunov.Slope.DataPoints[range - 1].Y);

                chart.AddTimeSeries("Lyapunov Function", lyapunov.Slope, SeriesChartType.Line);
                chart.AddTimeSeries("Sector", tsSector, SeriesChartType.Line, Color.Red);

                chart.ChartAreas[0].Axes[0].Title = "t";
                chart.ChartAreas[0].Axes[1].Title = "Slope";

                result = string.Format("{0:F5}",
                    (lyapunov.Slope.DataPoints[endPoint].Y - lyapunov.Slope.DataPoints[startPoint].Y) / (lyapunov.Slope.DataPoints[endPoint].X - lyapunov.Slope.DataPoints[startPoint].X)
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
            var mwSignalArray = new MWNumericArray(sourceData.TimeSeries.YValues);

            matlabEngine.Get2DWavelet(mwSignalArray, mwFolder, mwfileName, mwWname, tStart, tEnd, fStart, fEnd, dt, mwColMap, width, height);
        }

        public void DeleteTempFiles()
        {
            File.Delete(StringData.WaveletFile);
            File.Delete(StringData.WaveletPreviewFile);
        }
    }
}
