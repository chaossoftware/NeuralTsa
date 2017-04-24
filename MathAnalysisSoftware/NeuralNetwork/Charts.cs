using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Windows.Interop;
using System.Windows.Media.Imaging;

namespace NeuralNetwork {
    class Charts {

        private const float AOffset = 1f;
        private const float BOffset = 4f;
        private const float OutOffset = 7f;

        private static Font gridFont = new Font(new FontFamily("Cambria Math"), 13f);
        private static SolidBrush br = new SolidBrush(Color.White);

        private static double Max;
        private static double Min;
        private static int MaxIndexA;
        private static int MaxIndexB;
        private static int MinIndexA;
        private static int MinIndexB;

        public static Bitmap DrawNetworkState(int height, double[,] a, double[] b, int Dims, int Neurons, long iteration) {
            int maxItems = Math.Max(Dims, Neurons);
            Size itemSize = GetItemSize(height, maxItems);
            int width = (int)(itemSize.Height * 8d);

            Bitmap bitmap = new Bitmap(width, height);
            Graphics g = Graphics.FromImage(bitmap);
            GraphicsPath gp = new GraphicsPath();
            g.FillRectangle(new SolidBrush(Color.FromArgb(30, 30, 30)), new Rectangle(new Point(0, 0), new Size(width, height)));

            g.DrawString(String.Format("Neurons: {0} ; Dimensions: {1}\nIteration: {2:N0}", Neurons, Dims, iteration), gridFont, br, 0f, 0f);

            Max = GetMax(a);
            Min = GetMin(a);
            
            for (int i = 1; i <= Neurons; i++)
                for (int j = 1; j <= Dims; j++)
                    g.DrawLine(new Pen(GetColor(a[i,j]), 1.5f), GetItemCenter(i, Neurons, maxItems, itemSize.Width, BOffset), GetItemCenter(j, Dims, maxItems, itemSize.Width, AOffset));


            Max = GetMax(b);
            Min = GetMin(b);

            for (int i = 1; i <= Neurons; i++)
                g.DrawLine(new Pen(GetColor(b[i]), 1.5f), GetItemCenter(i, Neurons, maxItems, itemSize.Width, BOffset), GetItemCenter(1, 1, maxItems, itemSize.Width, OutOffset));


            for (int i = 1; i <= Dims; i++) {
                Rectangle rect = new Rectangle(GetItemUpLeft(i, Dims, maxItems, itemSize.Width, AOffset), itemSize);
                g.FillRectangle(new SolidBrush(Color.Orange), rect);
                gp.AddRectangle(rect);
            }

            for (int i = 1; i <= Neurons; i++) {
                Rectangle rect = new Rectangle(GetItemUpLeft(i, Neurons, maxItems, itemSize.Width, BOffset), itemSize);
                g.FillEllipse(new SolidBrush(Color.Red), rect);
                gp.AddEllipse(rect);
            }

            Rectangle outRect = new Rectangle(GetItemUpLeft(1, 1, maxItems, itemSize.Width, OutOffset), itemSize);
            g.FillEllipse(new SolidBrush(Color.Red), outRect);
            gp.AddEllipse(outRect);
            
            g.DrawPath(new Pen(Color.Black, 4), gp);

            gp.Dispose();
            g.Dispose();

            return bitmap;
        }



        #region "Entities"
        
        /// <summary>
        /// Get Entity center coordinates
        /// </summary>
        /// <param name="index">index of entity in list</param>
        /// <param name="items">count of entities in current list</param>
        /// <param name="maxItems">count of entities in the longest list</param>
        /// <param name="itemSize">calculated single item size</param>
        /// <param name="xOffset">offset by X coordinate (based on entity type)</param>
        /// <returns></returns>
        private static Point GetItemCenter(int index, int items, int maxItems, float itemSize, float xOffset) {
            //float i = (maxItems - items) / 2 + index;
            //return new Point((int)(xOffset * itemSize), (int)(itemSize * (1.5f * i - 1) + itemSize / 2 - itemSize / 4));
            index += (maxItems - items) / 2;
            return new Point((int)(xOffset * itemSize), (int)(index * 1.5 * itemSize));
        }


        /// <summary>
        /// Get Entity left upper corner coordinates based on it's center coordinates
        /// </summary>
        /// <param name="index">index of entity in list</param>
        /// <param name="items">count of entities in current list</param>
        /// <param name="maxItems">count of entities in the longest list</param>
        /// <param name="itemSize">calculated single item size</param>
        /// <param name="xOffset">offset by X coordinate (based on entity type)</param>
        /// <returns></returns>
        private static Point GetItemUpLeft(int index, int items, int maxItems, float itemSize, float xOffset) {
            Point pt = GetItemCenter(index, items, maxItems, itemSize, xOffset); 
            pt.X = (int)(pt.X - itemSize / 2);
            pt.Y = (int)(pt.Y - itemSize / 2);
            return pt;
        }


        /// <summary>
        /// Calculate entity size based on count of entities in list and specified imege height
        /// </summary>
        /// <param name="height">image height</param>
        /// <param name="itemsCount">count of entities in list</param>
        /// <returns></returns>
        private static Size GetItemSize(int height, int itemsCount) {
            int s = (int)((float)height / ((float)itemsCount + ((float)itemsCount / 2 + 0.5f)));
            return new Size(s, s);
        }

        #endregion


        #region "Synapses"

        private static double GetMax(Array a) {
            double max = double.MinValue;
            foreach (double aij in a)
                max = Math.Max(max, aij);

            return max;
        }


        private static double GetMin(Array a) {
            double min = double.MaxValue;
            foreach (double aij in a)
                min = Math.Min(min, aij);

            return min;
        }
               
        
        private static double GetWeightCoefficient(double current) {
            if (current < 0)
                return (int)(current / Min);
            else
                return (int)(current / Max);
        }


        /// <summary>
        /// Get synapse color
        /// </summary>
        /// <param name="current"></param>
        /// <returns></returns>
        private static Color GetColor(double current) {
            if (current == 0)
                return Color.FromArgb(0, 0, 0);

            if (current < 0) {
                return Color.FromArgb(0, 0, (int)(255 * Math.Abs(GetWeightCoefficient(current))));
            }
            else {
                return Color.FromArgb((int)(255 * GetWeightCoefficient(current)), (int)(255 * GetWeightCoefficient(current)), 0);
            }

        }

        #endregion



        #region "Neural Net animation"

        private static GifBitmapEncoder gEnc;
        
        public static void InitAnimation() {
            gEnc = new GifBitmapEncoder();
        }


        [System.Runtime.InteropServices.DllImport("gdi32.dll")]
        public static extern bool DeleteObject(IntPtr hObject);

        public static void AddFrame(Bitmap frame) {

            
                IntPtr hBitmap = frame.GetHbitmap();

                BitmapSource bitmap = Imaging.CreateBitmapSourceFromHBitmap(
                    hBitmap,
                    IntPtr.Zero,
                    System.Windows.Int32Rect.Empty,
                    BitmapSizeOptions.FromEmptyOptions());


                gEnc.Frames.Add(BitmapFrame.Create(bitmap));

                DeleteObject(hBitmap);

            /*var src = Imaging.CreateBitmapSourceFromHBitmap(
                    frame.GetHbitmap(),
                    IntPtr.Zero,
                    System.Windows.Int32Rect.Empty,
                    BitmapSizeOptions.FromEmptyOptions());
                gEnc.Frames.Add(BitmapFrame.Create(src));*/
        }



        public static void SaveAnimation() {
            using (FileStream fs = new FileStream(NeuralOutput.BasePath + "_neural_anim.gif", FileMode.Create)) {
                gEnc.Save(fs);
            }
        }

        #endregion
    }
}