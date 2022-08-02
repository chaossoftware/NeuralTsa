using System.Drawing;

namespace ChaosSoft.DrawEngine.ColorMaps
{
    public enum ColorMap
    {
        Orange,
        Parula
    }

    public interface IColorMap
    {
        Color GetColor(double value);
    }
}
