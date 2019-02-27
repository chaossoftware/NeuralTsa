using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;
using MathLib.DrawEngine.Charts;

namespace TimeSeriesAnalysis
{
    public partial class PreviewForm : Form
    {
        public PreviewForm(string caption)
        {
            InitializeComponent();
            Text = caption;
        }

        public PreviewForm(string caption, int width, int height)
        {
            InitializeComponent();
            Text = caption;

            int diffWidth = this.previewPBox.Width - width;
            int diffHeight = this.previewPBox.Height - height;

            this.Width -= diffWidth;
            this.Height -= diffHeight;
        }

        public PlotObject Plot { get; set; }

        private void Redraw()
        {
            if (Plot != null)
            {
                Plot.Size = new Size(previewPBox.Width, previewPBox.Height);
                previewPBox.Image = Plot.Plot();
            }
        }

        private void PreviewContextMenu(object sender, MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Right)
                contextMenu.Show(previewPBox, new Point(e.X, e.Y));
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

            if (previewPBox.Image != null)
                previewPBox.Image.Save(fileName, ImageFormat.Png);
        }

        private void previewPBox_SizeChanged(object sender, EventArgs e) =>
            Redraw();

        private void copyItem_Click(object sender, EventArgs e) =>
            Clipboard.SetImage(previewPBox.Image);
    }
}
