using System;
using System.Drawing;
using System.Windows.Forms;
using MathLib.DrawEngine.Charts;

namespace TimeSeriesAnalysis
{
    public partial class PreviewForm : Form
    {

        public PlotObject plotObject;

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


        private void redraw() {
            if (plotObject != null) {
                plotObject.Size = new Size(previewPBox.Width, previewPBox.Height);
                previewPBox.Image = plotObject.Plot();
            }
        }


        private void PreviewContextMenu(object sender, MouseEventArgs e) {
            if (e.Button == MouseButtons.Right) {
                contextMenu.Show(previewPBox, new Point(e.X, e.Y));
            }
        }


        private void ContextMenuClick(object sender, EventArgs e) {
            SaveFileDialog saveDialog = new SaveFileDialog();
            saveDialog.AddExtension = true;
            saveDialog.Filter = "Png image|*.png";
            saveDialog.ShowDialog();
            savePreview(saveDialog.FileName);
        }


        private void savePreview(string fileName) {
            if (fileName.Equals("")) {
                MessageBox.Show("Выберите путь для сохранения.");
                return;
            }
            if (previewPBox.Image != null) {
                previewPBox.Image.Save(fileName, System.Drawing.Imaging.ImageFormat.Png);
            }
        }

        private void previewPBox_SizeChanged(object sender, EventArgs e) {
            redraw();
        }

        private void copyItem_Click(object sender, EventArgs e)
        {
            Clipboard.SetImage(previewPBox.Image);
        }
    }
}
