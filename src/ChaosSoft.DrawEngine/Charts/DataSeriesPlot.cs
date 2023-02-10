using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using ChaosSoft.Core.Data;

namespace ChaosSoft.DrawEngine.Charts
{
    /// <summary>
    /// Class for Multi Signal plot
    /// </summary>
    public abstract class DataSeriesPlot : PlotObject
    {
        public DataSeriesPlot(Size bitmapSize) 
            : base(bitmapSize, 1f)
        {
            this.TimeSeriesList = new List<DataSeries>();
            this.PlotPens = new List<Pen>();
            this.tsAmplitude = new DataPoint(0, 0);
            this.tsPointMax = new DataPoint(double.MinValue, double.MinValue);
            this.tsPointMin = new DataPoint(double.MaxValue, double.MaxValue);
        }

        public DataSeriesPlot(Size bitmapSize, DataSeries dataSeries, Color color, float thickness) 
            : this(bitmapSize)
        {
            this.AddDataSeries(dataSeries, color, thickness);
        }

        public DataSeriesPlot(Size bitmapSize, DataSeries dataSeries) 
            : this(bitmapSize, dataSeries, Color.SteelBlue, 1f)
        {
        }

        protected List<DataSeries> TimeSeriesList { get; set; }

        protected List<Pen> PlotPens { get; set; }

        protected DataPoint tsPointMax;

        protected DataPoint tsPointMin;

        protected DataPoint tsAmplitude;

        public void AddDataSeries(DataSeries dataSeries, Color color, float thickness)
        {
            TimeSeriesList.Add(dataSeries);
            PlotPens.Add(new Pen(color, thickness));

            tsPointMax = new DataPoint(Math.Max(tsPointMax.X, dataSeries.Max.X), Math.Max(tsPointMax.Y, dataSeries.Max.Y));
            tsPointMin = new DataPoint(Math.Min(tsPointMin.X, dataSeries.Min.X), Math.Min(tsPointMin.Y, dataSeries.Min.Y));
            tsAmplitude = new DataPoint(tsPointMax.X - tsPointMin.X, tsPointMax.Y - tsPointMin.Y);
        }

        public void AddDataSeries(DataSeries dataSeries, Color color) =>
            AddDataSeries(dataSeries, color, 1f);

        public override Bitmap Plot()
        {
            PrepareChartArea();

            if (TimeSeriesList.All(ts => ts.Length < 1))
            {
                NoDataToPlot();
            }
            else
            {
                CalculateChartAreaSize(this.tsAmplitude);

                for (int i = 0; i < this.TimeSeriesList.Count; i++)
                {
                    DrawDataSeries(this.TimeSeriesList[i], this.PlotPens[i]);
                }

                DrawGrid();
            }

            g.Dispose();

            return PlotBitmap;
        }

        protected override void DrawGrid()
        {
            SetAxisValues(
                GetAxisValue(tsPointMin.X),
                GetAxisValue(tsPointMax.X),
                GetAxisValue(tsPointMin.Y),
                GetAxisValue(tsPointMax.Y)
            );
        }

        protected abstract void DrawDataSeries(DataSeries ds, Pen pen);
    }
}
