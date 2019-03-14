using System;
using System.Drawing;
using System.IO;
using MathLib;
using MathLib.Data;
using MathLib.DrawEngine.Charts;
using MathLib.MathMethods.Lyapunov;
using MathLib.Transform;
using MathWorks.MATLAB.NET.Arrays;

namespace TimeSeriesAnalysis
{
    internal class Routines
    {
        public SourceData sourceData;

        public LyapunovMethod lyapunov;

        public SignalPlot GetSignalPlot(Size size, float thickness, bool withTime, int startPoint, int endPoint) =>
            new SignalPlot(sourceData.TimeSeries, size, thickness);
        
        public MapPlot GetPoincarePlot(Size size, int thickness) =>
            new MapPlot(Ext.GeneratePseudoPoincareMapData(sourceData.TimeSeries.YValues), size, thickness);

        public PlotObject GetLyapunovPlot(Size size, int thickness, int startPoint, int endPoint, bool isWolf, out string result)
        {
            PlotObject lyap;
            int range = endPoint - startPoint + 1;
            result = string.Empty;
                
            if (isWolf)
            {
                var plotSeries = new Timeseries();

                for (int i = startPoint; i < range; i++)
                {
                    plotSeries.AddDataPoint(lyapunov.slope.DataPoints[i].X, lyapunov.slope.DataPoints[i].Y);
                }

                lyap = new SignalPlot(plotSeries, size, 1);
                lyap.LabelY = "LE";
            }
            else
            {
                lyap = new MultiSignalPlot(size);
                ((MultiSignalPlot)lyap).AddDataSeries(lyapunov.slope, Color.SteelBlue);
                var markerSeries = new Timeseries();
                markerSeries.AddDataPoint(lyapunov.slope.DataPoints[startPoint].X, lyapunov.slope.DataPoints[startPoint].Y);
                markerSeries.AddDataPoint(lyapunov.slope.DataPoints[range - 1].X, lyapunov.slope.DataPoints[range - 1].Y);
                ((MultiSignalPlot)lyap).AddDataSeries(markerSeries, Color.Red);
                lyap.LabelY = "Slope";
                lyap.LabelX = "t";

                result = string.Format("{0:F5}",
                    (lyapunov.slope.DataPoints[endPoint].Y - lyapunov.slope.DataPoints[startPoint].Y) / (lyapunov.slope.DataPoints[endPoint].X - lyapunov.slope.DataPoints[startPoint].X)
                );
            }
            return lyap;
        }

        public SignalPlot GetFourierPlot(Size size, int thickness, double statrFreq, double endFreq, double dt, bool logScale)
        {
            var fourierSeries = Fourier.GetFourier(sourceData.TimeSeries.YValues, statrFreq, endFreq, dt, Convert.ToInt32(logScale));

            return new SignalPlot(fourierSeries, size, thickness)
            {
                LabelY = "F(ω)",
                LabelX = "ω"
            };
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
