namespace NeuralNetTsa.Configuration
{
    public class DataFile
    {
        public DataFile(string fileName, int dataColumn, int points, int startPoint, int endPoint)
        {
            FileName = fileName;
            DataColumn = dataColumn;
            StartPoint = startPoint;
            EndPoint = endPoint;
            Points = points;
        }

        public string FileName { get; }

        public int DataColumn { get; }

        public int StartPoint { get; }

        public int EndPoint { get; }

        public int Points { get; }

        public OutputParameters Output { get; set; }
    }
}
