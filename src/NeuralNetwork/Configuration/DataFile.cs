namespace NeuralAnalyser.Configuration
{
    public class DataFile
    {
        public DataFile(string fileName, int dataColumn, int points, int startPoint, int endPoint)
        {
            this.FileName = fileName;
            this.DataColumn = dataColumn;
            this.StartPoint = startPoint;
            this.EndPoint = endPoint;
            this.Points = points;
        }

        public string FileName { get; set; }

        public int DataColumn { get; set; }

        public int StartPoint { get; set; }

        public int EndPoint { get; set; }

        public int Points { get; set; }

        public OutputParameters Output { get; set; }
    }
}
