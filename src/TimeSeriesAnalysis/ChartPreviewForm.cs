using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;

namespace TimeSeriesAnalysis
{
    public partial class ChartPreviewForm : Form
    {
        public ChartPreviewForm(string caption)
        {
            InitializeComponent();
            Text = caption;
        }

        public ChartPreviewForm(string caption, int width, int height)
        {
            InitializeComponent();
            Text = caption;

            int diffWidth = this.chart.Width - width;
            int diffHeight = this.chart.Height - height;

            this.Width -= diffWidth;
            this.Height -= diffHeight;
        }

        private void ContextMenuClick(object sender, EventArgs e)
        {
            var saveDialog = new SaveFileDialog()
            {
                AddExtension = true,
                Filter = "Png image|*.png"
            };

            saveDialog.ShowDialog();
            SavePreview(saveDialog.FileName);
        }

        private void SavePreview(string fileName)
        {
            if (string.IsNullOrEmpty(fileName))
            {
                MessageBox.Show("Select path to save.");
                return;
            }

            if (chart.Series[0].Points.Count != 0)
            {
                chart.SaveImage(fileName, ImageFormat.Png);
            }
        }

        private void copyItem_Click(object sender, EventArgs e) =>
            Clipboard.SetImage(GetBitmap());

        private void chart_MouseClick(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Right)
                contextMenu.Show(chart, new Point(e.X, e.Y));
        }

        private Bitmap GetBitmap()
        {
            var bitmap = new Bitmap(chart.Width, chart.Height);
            chart.DrawToBitmap(bitmap, new Rectangle(0, 0, chart.Width, chart.Height));

            return bitmap;
        }
    }
}
