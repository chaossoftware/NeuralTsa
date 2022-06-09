namespace NeuralNetTsa.Routines
{
    internal class AttractorData
    {
        public AttractorData(double[] xt, double[] yt, double[] zt)
        {
            Xt = xt; 
            Yt = yt; 
            Zt = zt; 
        }

        public double[] Xt { get; }

        public double[] Yt { get; }

        public double[] Zt { get; }
    }
}
