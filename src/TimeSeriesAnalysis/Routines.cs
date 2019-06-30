using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
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

        public void FillSignalChart(MathChart chart) =>
            chart.ClearChart()
            .SetAxisNames("t", "f(t)")
            .AddTimeSeries("Signal", sourceData.TimeSeries, SeriesChartType.Line);

        public void FillPoincareChart(MathChart chart) =>
            chart.ClearChart()
            .SetAxisNames("f(t)", "f(t+1)")
            .AddTimeSeries("Pseudo Poincare Section", PseudoPoincareMap.GetMapDataFrom(sourceData.TimeSeries.YValues), SeriesChartType.Point);

        public void FillFourierChart(MathChart chart, double statrFreq, double endFreq, double dt, bool logScale)
        {
            try
            {
                var timeseries = Fourier.GetFourier(sourceData.TimeSeries.YValues, statrFreq, endFreq, dt, Convert.ToInt32(logScale));
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
                    timeseries.AddDataPoint(lyapunov.Slope.DataPoints[i].X, lyapunov.Slope.DataPoints[i].Y);
                }

                chart.SetAxisNames("t", "LE")
                    .AddTimeSeries("Lyapunov Exponent in Time", timeseries, SeriesChartType.Line);
            }
            else
            {
                var tsSector = new Timeseries();

                tsSector.AddDataPoint(lyapunov.Slope.DataPoints[startPoint].X, lyapunov.Slope.DataPoints[startPoint].Y);
                tsSector.AddDataPoint(lyapunov.Slope.DataPoints[range - 1].X, lyapunov.Slope.DataPoints[range - 1].Y);

                chart.SetAxisNames("t", "Slope")
                    .AddTimeSeries("Lyapunov Function", lyapunov.Slope, SeriesChartType.Line)
                    .AddTimeSeries("Sector", tsSector, SeriesChartType.Line, Color.Red);

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

        public int SlopeChangePointIndex(Timeseries timeseries, int groupingCoefficient, double cutOffValue)
        {
            var smoothed = new List<PointF>();     // reduced smoothed data
            var d1 = new List<PointF>();     // 1st derivative
            var d2 = new List<PointF>();     // 2nd derivative
            var m = new List<PointF>();      // reasonably large values from D2

            // smoothen the data
            for (int i = 1; i < timeseries.Length / groupingCoefficient; i++)
            {
                double ysum = 0.0f;

                for (int j = 0; j < groupingCoefficient; j++)
                {
                    ysum += timeseries.DataPoints[i * groupingCoefficient + j].Y;
                }

                smoothed.Add(new PointF(i, (float)ysum / groupingCoefficient));
            }

            // 1st derivative
            for (int i = 1; i < smoothed.Count; i++)
            {
                d1.Add(new PointF(i, smoothed[i - 1].Y - smoothed[i].Y));
            }

            // 2nd derivative
            for (int i = 1; i < d1.Count; i++)
            {
                d2.Add(new PointF(i, d1[i - 1].Y - d1[i].Y));
            }

            // collect 'reasonably' large values from D2
            foreach (var p in d2.Where(p => Math.Abs(p.Y / cutOffValue) > 1))
            {
                m.Add(p);
            }

            return m.Any() ? (int)(m.Last().X * groupingCoefficient) /*+ groupingCoefficient*/ : 0;
        }
    }
}
