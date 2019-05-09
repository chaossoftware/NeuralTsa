using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Threading;
using System.Windows.Forms;
using MathLib.IO;
using MathLib.MathMethods.Lyapunov;
using System.Globalization;
using MathLib.Data;

namespace TimeSeriesAnalysis {
    public partial class MainForm : Form
    {
        private Routines routines = new Routines();

        public MainForm()
        {
            InitializeComponent();
        }

        //File open and read
        private void openFileBtn_Click(object sender, EventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "All files|*.*|Time series data|*.dat *.txt *.csv";
            openFileDialog.ShowDialog();
            string fName = openFileDialog.FileName;

            if (string.IsNullOrEmpty(fName))
                return;

            CleanUp();

            try
            {
                routines.sourceData = new SourceData(fName);
                FillUiWithData();
            }
            catch (ArgumentException ex)
            {
                MessageBox.Show("Unable to read file file:" + ex.Message);
            }
        }

        //Perform calculation in a separate thread
        private void startBtn_Click(object sender, EventArgs e)
        {
            if (RefreshTimeSeries())
            {
                le_resultText.Text = StringData.Calculating;
                le_resultText.BackColor = Color.OrangeRed;

                new Thread(CalculateLyapunovExponent)
                    .Start();
            }
        }

        private bool RefreshTimeSeries()
        {
            if (routines.sourceData == null)
            {
                MessageBox.Show(StringData.MsgEmptyFile);
                return false;
            }

            routines.sourceData.SetTimeSeries(
                sourceColumnNum.ToInt() - 1,
                startPointNum.ToInt(),
                endPointNum.ToInt(),
                pointsNum.ToInt(),
                useTimeCheckbox.Checked
            );

            return true;
        }

        private void saveBtn_Click(object sender, EventArgs e)
        {
            if (routines.sourceData == null)
            {
                MessageBox.Show(StringData.MsgEmptyFile);
                return;
            }

            var outDir = Path.Combine(routines.sourceData.Folder, routines.sourceData.FileName + "_rez");
            string fName = Path.Combine(outDir, routines.sourceData.FileName);

            if (!Directory.Exists(outDir))
                Directory.CreateDirectory(outDir);

            if (chartSignal.Series[0].Points.Count != 0 && chartPoincare.Series[0].Points.Count != 0)
            {
                routines.SaveChart(chartSignal, fName + "_plot");
                routines.SaveChart(chartPoincare, fName + "_poincare");
                DataWriter.CreateDataFile(fName + "_signal", routines.sourceData.GetTimeSeriesString(false));
            }

            if (chartFft.Series[0].Points.Count != 0)
            {
                routines.SaveChart(chartFft, fName + "_fourier");
            }

            if (wav_plotPBox.Image != null)
                wav_plotPBox.Image.Save(fName + "_wavelet.png", ImageFormat.Png);

            if (routines.lyapunov != null)
                DataWriter.CreateDataFile(fName + "_lyapunov.txt", routines.lyapunov.GetInfoFull());

            if (chartLyapunov.Series[0].Points.Count != 0)
            {
                routines.SaveChart(chartLyapunov, fName + "_lyapunovSlope");
            }
        }

        private void FillUiWithData()
        {
            fileNameLbl.Text = routines.sourceData.ToString().Replace("\n", " ");

            sourceColumnNum.Maximum = routines.sourceData.ColumnsCount;
            sourceColumnNum.Minimum = 1;

            startPointNum.Maximum = routines.sourceData.Length - 1;

            endPointNum.Maximum = routines.sourceData.Length;
            endPointNum.Value = routines.sourceData.Length;
        }

        #region "CHARTS"

        private void plotBtn_Click(object sender, EventArgs e)
        {
            if (!RefreshTimeSeries())
                return;

            routines.DeleteTempFiles();

            routines.FillSignalChart(this.chartSignal);
            routines.FillPoincareChart(this.chartPoincare);

            if (fourierCheckbox.Checked)
            {
                routines.FillFourierChart(chartFft,
                    fft_fStartNum.ToDouble(),
                    fft_fEndNum.ToDouble(),
                    fft_dtNum.ToDouble(),
                    fft_logCbox.Checked);
            }

            if (waveletCheckbox.Checked)
            {
                wav_plotPBox.Image = null;
                double tStart = routines.sourceData.TimeSeries.Min.X;
                double tEnd = routines.sourceData.TimeSeries.Max.X;

                try
                {
                    routines.BuildWavelet(StringData.WaveletFile, wav_nameCombo.Text, tStart, tEnd, wav_fStartNum.ToDouble(), wav_fEndNum.ToDouble(), wav_dtNum.ToDouble(), wav_colorMapCombo.Text, wav_plotPBox.Width, wav_plotPBox.Height);
                    GetImageFromFile(wav_plotPBox, StringData.WaveletFile);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Unable to build {StringData.Wavelet}:\n" + ex.Message);
                }
            }
        }

        private void lyapunovRedrawBtn_Click(object sender, EventArgs e) {
            try
            {

                if (routines.lyapunov is KantzMethod)
                    ((KantzMethod)routines.lyapunov).SetSlope(this.le_kantz_slopeCombo.Text);

                var res = routines.FillLyapunovChart(chartLyapunov, le_pStartNum.ToInt(), le_pEndNum.ToInt(), le_wolf_radio.Checked);

                if (routines.lyapunov is KantzMethod || routines.lyapunov is RosensteinMethod)
                    le_resultText.Text = res;
            }
            catch (Exception ex) {
                MessageBox.Show("Error plotting Lyapunov slope: " + ex.Message); 
            }
        }

        private void waveletPBox_DoubleClick(object sender, EventArgs e)
        {
            var pf = GetPreviewForm(this.wav_nameCombo.Text + " " + StringData.Wavelet);

            try
            {
                routines.BuildWavelet(
                    StringData.WaveletPreviewFile, 
                    wav_nameCombo.Text,
                    routines.sourceData.TimeSeries.Min.X,
                    routines.sourceData.TimeSeries.Max.X, 
                    wav_fStartNum.ToDouble(), 
                    wav_fEndNum.ToDouble(), 
                    wav_dtNum.ToDouble(), 
                    wav_colorMapCombo.Text, 
                    wav_plotPBox.Width, 
                    wav_plotPBox.Height);

                GetImageFromFile(pf.previewPBox, StringData.WaveletPreviewFile);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Unable to build {StringData.Fourier}:\n" + ex.Message);
            }
        }

        private void GetImageFromFile(PictureBox pb, string fileName)
        {
            using (var fs = new FileStream(fileName, FileMode.Open, FileAccess.Read))
            {
                pb.Image = Image.FromStream(fs);
                pb.SizeMode = PictureBoxSizeMode.StretchImage;
            }
        }

        #endregion

        #region lyapunovRelated

        private void CalculateLyapunovExponent()
        {
            int dim = le_dimNum.ToInt();
            int tau = le_tauNum.ToInt();
            double scaleMin = le_scaleMinNum.ToDouble();

            if (le_wolf_radio.Checked)
                routines.lyapunov = new WolfMethod(
                    routines.sourceData.TimeSeries.YValues,
                    dim,
                    tau,
                    le_wolf_stepNum.ToDouble(),
                    scaleMin,
                    le_scaleMaxNum.ToDouble(), 
                    le_wolf_evolveStepsNum.ToInt()
                );

            if (lyap_calc_Rad_rosenstein.Checked)
                routines.lyapunov = new RosensteinMethod(
                    routines.sourceData.TimeSeries.YValues,
                    dim,
                    tau, 
                    le_ros_stepsNum.ToInt(), 
                    le_ros_distanceNum.ToInt(),
                    scaleMin
                );

            if (lyap_calc_Rad_kantz.Checked)
                routines.lyapunov = new KantzMethod(
                    routines.sourceData.TimeSeries.YValues,
                    dim,
                    tau,
                    le_kantz_maxIterNum.ToInt(),
                    le_kantz_windowNum.ToInt(),
                    scaleMin,
                    le_scaleMaxNum.ToDouble(),
                    le_kantz_scalesNum.ToInt()
                );

            try
            {
                routines.lyapunov.Calculate();
                var mi = new MethodInvoker(this.SetLyapunovResult);
                this.BeginInvoke(mi);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Unable to calculate LE:\n" + ex.Message);
                le_resultText.Text = "Error";
            }
        }

        private void SetLyapunovResult()
        {
            le_resultText.BackColor = Color.Khaki;
            string result = string.Empty;

            if (routines.lyapunov is WolfMethod)
            {
                result = string.Format("{0:F5}", ((WolfMethod)routines.lyapunov).rezult);
                le_resultText.Text = result;
            }

            if (routines.lyapunov is KantzMethod)
            {
                this.le_kantz_slopeCombo.Items.Clear();
                string[] items = new string[((KantzMethod)routines.lyapunov).SlopesList.Count];
                ((KantzMethod)routines.lyapunov).SlopesList.Keys.CopyTo(items, 0);
                this.le_kantz_slopeCombo.Items.AddRange(items);
                this.le_kantz_slopeCombo.SelectedIndex = 0;
                ((KantzMethod)routines.lyapunov).SetSlope(this.le_kantz_slopeCombo.Text);
            }

            if (routines.lyapunov.Slope.Length > 1)
            {
                le_resultText.Text = routines.lyapunov.GetInfoShort();
                le_pEndNum.Value = routines.lyapunov.Slope.Length - 1;

                try
                {
                    routines.FillLyapunovChart(chartLyapunov, le_pStartNum.ToInt(), le_pEndNum.ToInt(), le_wolf_radio.Checked);
                }
                catch (Exception ex)
                {
                    MessageBox.Show("Error plotting Lyapunov slope: " + ex.Message);
                    result = StringData.NoValue;
                }
            }
            else
            {
                result = StringData.NoValue;
            }
               
            if (routines.lyapunov is KantzMethod || routines.lyapunov is RosensteinMethod)
            {
                le_resultText.Text = result;
            }
        }

        #endregion

        private void useTimeCheckbox_CheckedChanged(object sender, EventArgs e)
        {
            if (this.useTimeCheckbox.Checked)
            {
                RefreshTimeSeries();
                sourceStepTxt.Text = string.Format(CultureInfo.InvariantCulture, "{0:F8}", routines.sourceData.Step);
            }
            else
            {
                sourceStepTxt.Text = string.Empty;
            }
        }

        private void CleanUp()
        {
            routines.sourceData = null;
            chartSignal.Series[0].Points.Clear();
            chartPoincare.Series[0].Points.Clear();
            chartFft.Series[0].Points.Clear();

            foreach (var series in chartLyapunov.Series)
            {
                series.Points.Clear();
            }
            
            wav_plotPBox.Image = null;
            routines.lyapunov = null;
            routines.DeleteTempFiles();
        }

        private void mainForm_FormClosing(object sender, FormClosingEventArgs e) =>
            CleanUp();

        private void pointsNum_ValueChanged(object sender, EventArgs e)
        {
            if (this.useTimeCheckbox.Checked)
            {
                RefreshTimeSeries();
                sourceStepTxt.Text = string.Format(CultureInfo.InvariantCulture, "{0:F8}", routines.sourceData.Step);
            }
        }

        private PreviewForm GetPreviewForm(string title)
        {
            var form = new PreviewForm(
                title,
                this.numPreviewWidth.ToInt(),
                this.numPreviewHeight.ToInt());

            form.Show();

            return form;
        }

        private ChartPreviewForm GetChartPreviewForm(string title)
        {
            var form = new ChartPreviewForm(
                title,
                this.numPreviewWidth.ToInt(),
                this.numPreviewHeight.ToInt());

            form.Show();

            return form;
        }

        private void chartSignal_DoubleClick(object sender, EventArgs e)
        {
            if (chartSignal.Series[0].Points.Count != 0)
            {
                var pf = GetChartPreviewForm(StringData.Signal);
                routines.FillSignalChart(pf.chart);
            }
        }

        private void chartPoincare_DoubleClick(object sender, EventArgs e)
        {
            if (chartPoincare.Series[0].Points.Count != 0)
            {
                var pf = GetChartPreviewForm(StringData.Poincare);
                routines.FillPoincareChart(pf.chart);
            }
        }

        private void chartFft_DoubleClick(object sender, EventArgs e)
        {
            if (chartFft.Series[0].Points.Count != 0)
            {
                var pf = GetChartPreviewForm(StringData.Signal);
                routines.FillFourierChart(pf.chart,
                    fft_fStartNum.ToDouble(),
                    fft_fEndNum.ToDouble(),
                    fft_dtNum.ToDouble(),
                    fft_logCbox.Checked);
            }
        }
    }
}
