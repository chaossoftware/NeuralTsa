using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using MathLib.Data;
using MathLib.MathMethods.Lyapunov;
using MathLib.Transform;
using MathWorks.MATLAB.NET.Arrays;

namespace TimeSeriesAnalysis
{
    internal class Routines
    {
        public SourceData sourceData;

        public LyapunovMethod lyapunov;

        public void FillSignalChart(Chart chart)
        {
            chart.Series[0].ChartType = SeriesChartType.Line;
            chart.ChartAreas[0].Axes[0].Title = "t";
            chart.ChartAreas[0].Axes[1].Title = "f(t)";
            chart.Series[0].Points.Clear();

            foreach (var dp in sourceData.TimeSeries.DataPoints)
            {
                chart.Series[0].Points.AddXY(dp.X, dp.Y);
            }
        }

        public void FillFourierChart(Chart chart, double statrFreq, double endFreq, double dt, bool logScale)
        {
            try
            {
                chart.Series[0].ChartType = SeriesChartType.Line;
                chart.ChartAreas[0].Axes[0].Title = "ω";
                chart.ChartAreas[0].Axes[1].Title = "f(ω)";
                chart.Series[0].Points.Clear();
                var dataPoints = Fourier.GetFourier(sourceData.TimeSeries.YValues, statrFreq, endFreq, dt, Convert.ToInt32(logScale)).DataPoints;

                foreach (var dp in sourceData.TimeSeries.DataPoints)
                {
                    chart.Series[0].Points.AddXY(dp.X, dp.Y);
                }
            }
            catch (Exception ex)
            {
                chart.Series[0].Points.Clear();
                MessageBox.Show($"Unable to build {StringData.Fourier}:\n" + ex.Message);
            }
        }

        public void FillPoincareChart(Chart chart)
        {
            chart.Series[0].ChartType = SeriesChartType.Point;
            chart.ChartAreas[0].Axes[0].Title = "f(t)";
            chart.ChartAreas[0].Axes[1].Title = "f(t+1)";
            chart.Series[0].Points.Clear();
            var dataPoints = PseudoPoincareMap.GetMapDataFrom(sourceData.TimeSeries.YValues).DataPoints;

            foreach (var dp in dataPoints)
            {
                chart.Series[0].Points.AddXY(dp.X, dp.Y);
            }
        }

        public string FillLyapunovChart(Chart chart, int startPoint, int endPoint, bool isWolf)
        {
            chart.Series[0].ChartType = SeriesChartType.Line;

            foreach (var series in chart.Series)
            {
                series.Points.Clear();
            }

            int range = endPoint - startPoint + 1;
            var result = string.Empty;

            if (isWolf)
            {
                chart.ChartAreas[0].Axes[0].Title = "t";
                chart.ChartAreas[0].Axes[1].Title = "LE";

                for (int i = startPoint; i < range; i++)
                {
                    chart.Series[0].Points.AddXY(lyapunov.Slope.DataPoints[i].X, lyapunov.Slope.DataPoints[i].Y);
                }
            }
            else
            {
                chart.ChartAreas[0].Axes[0].Title = "t";
                chart.ChartAreas[0].Axes[1].Title = "Slope";

                foreach (var dp in lyapunov.Slope.DataPoints)
                {
                    chart.Series[0].Points.AddXY(dp.X, dp.Y);
                }

                chart.Series[1].Points.AddXY(lyapunov.Slope.DataPoints[startPoint].X, lyapunov.Slope.DataPoints[startPoint].Y);
                chart.Series[1].Points.AddXY(lyapunov.Slope.DataPoints[range - 1].X, lyapunov.Slope.DataPoints[range - 1].Y);

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

        public void SaveChart(Chart chart, string filePath)
        {
            var bitmap = new Bitmap(chart.Width, chart.Height);
            chart.DrawToBitmap(bitmap, new Rectangle(0, 0, chart.Width, chart.Height));

            bitmap.Save(filePath + "." + ImageFormat.Png, ImageFormat.Png);
        }
    }
}
