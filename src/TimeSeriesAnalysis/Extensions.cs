using System;
using System.Globalization;
using System.Windows.Forms;

namespace TimeSeriesAnalysis
{
    public static class Extensions
    {
        public static double ToDouble(this NumericUpDown num) =>
            Convert.ToDouble(num.Value, CultureInfo.InvariantCulture);

        public static int ToInt(this NumericUpDown num) =>
            Convert.ToInt32(num.Value);
    }
}
