using MathLib;
using MathLib.Data;
using MathLib.DrawEngine;
using MathLib.DrawEngine.Charts;
using MathLib.IO;
using MathLib.MathMethods.Lyapunov;
using MathLib.Transform;
using MathWorks.MATLAB.NET.Arrays;
using System.Drawing;
using System.IO;

namespace TimeSeriesAnalysis
{
    class Routines {

        public SourceData sourceData;

        public LyapunovMethod lyapunov;


        public SignalPlot GetSignalPlot(Size size, float thickness, bool withTime, int startPoint, int endPoint) {
            return new SignalPlot(sourceData.TimeSeries, size, thickness);
        }


        public MapPlot GetPoincarePlot(Size size, int thickness) {
            return new MapPlot(Ext.GeneratePseudoPoincareMapData(sourceData.TimeSeries.YValues), size, thickness);
        }


        public PlotObject GetLyapunovPlot(Size size, int thickness, int startPoint, int endPoint, bool isWolf, out string result) {
            PlotObject lyap;
            int range = endPoint - startPoint + 1;
            result = "";
                
            if (isWolf)
            {
                Timeseries plotSeries = new Timeseries();

                for (int i = startPoint; i < range; i++)
                {
                    plotSeries.AddDataPoint(lyapunov.slope.DataPoints[i].X, lyapunov.slope.DataPoints[i].Y);
                }

                lyap = new SignalPlot(plotSeries, size, 1);
                lyap.LabelY = "LE";
            }
            else
            {
                lyap = new MultiSignalPlot(size, 1);
                ((MultiSignalPlot)lyap).AddDataSeries(lyapunov.slope, Color.SteelBlue);
                Timeseries markerSeries = new Timeseries();
                markerSeries.AddDataPoint(lyapunov.slope.DataPoints[startPoint].X, lyapunov.slope.DataPoints[startPoint].Y);
                markerSeries.AddDataPoint(lyapunov.slope.DataPoints[range - 1].X, lyapunov.slope.DataPoints[range - 1].Y);
                ((MultiSignalPlot)lyap).AddDataSeries(markerSeries, Color.Red);
                lyap.LabelY = "Slope";

                result = string.Format("{0:F5}",
                    (lyapunov.slope.DataPoints[endPoint].Y - lyapunov.slope.DataPoints[startPoint].Y) / (lyapunov.slope.DataPoints[endPoint].X - lyapunov.slope.DataPoints[startPoint].X)
                );
            }
            return lyap;
        }


        public SignalPlot GetFourierPlot(Size size, int thickness, double statrFreq, double endFreq, double dt, int logScale) {

            Timeseries fourierSeries = Fourier.GetFourier(sourceData.TimeSeries.YValues, statrFreq, endFreq, dt, logScale);
            var fourierPlot = new SignalPlot(fourierSeries, size, thickness);
            fourierPlot.LabelY = "F(ω)";
            fourierPlot.LabelX = "ω";
            return fourierPlot;
        }


        public void BuildWavelet(string tmpFileName, string wName, double tStart, double tEnd, double startFreq, double endFreq, double dt, string colMap, double width, double height)
        {
            MatlabEngine.MatlabEngine signalAnalysis = new MatlabEngine.MatlabEngine();
            MWCharArray mw_Folder = new MWCharArray("");
            MWCharArray mw_fileName = new MWCharArray(tmpFileName);
            MWCharArray mw_wname = new MWCharArray(wName);
            MWCharArray mw_colMap = new MWCharArray(colMap);
            MWNumericArray mw_signalArray = new MWNumericArray(sourceData.TimeSeries.YValues);

            signalAnalysis.Get2DWavelet(mw_signalArray, mw_Folder, mw_fileName, mw_wname, tStart, tEnd, startFreq, endFreq, dt, mw_colMap, width, height);
        }


        public void deleteTempFiles() {
            File.Delete("wavelet.tmp");
        }
    }
}
