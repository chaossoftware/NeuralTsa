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
            var openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "All files|*.*|Time series data|*.dat *.txt *.csv";
            openFileDialog.ShowDialog();
            string fName = openFileDialog.FileName;

            if (string.IsNullOrEmpty(fName))
            {
                return;
            }

            try
            {
                CleanUp();
                routines.SourceData = new SourceData(fName);
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
            if (routines.SourceData == null)
            {
                MessageBox.Show(StringData.MsgEmptyFile);
                return false;
            }

            routines.SourceData.SetTimeSeries(
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
            if (routines.SourceData == null)
            {
                MessageBox.Show(StringData.MsgEmptyFile);
                return;
            }

            var outDir = Path.Combine(routines.SourceData.Folder, routines.SourceData.FileName + "_rez");
            string fName = Path.Combine(outDir, routines.SourceData.FileName);

            if (!Directory.Exists(outDir))
            {
                Directory.CreateDirectory(outDir);
            }

            if (chartSignal.HasData && chartPoincare.HasData)
            {
                chartSignal.SaveImage(fName + "_plot", ImageFormat.Png);
                chartPoincare.SaveImage(fName + "_poincare", ImageFormat.Png);
                DataWriter.CreateDataFile(fName + "_signal", routines.SourceData.GetTimeSeriesString(false));
            }

            if (chartFft.HasData)
            {
                chartFft.SaveImage(fName + "_fourier", ImageFormat.Png);
            }

            if (wav_plotPBox.Image != null)
            {
                wav_plotPBox.Image.Save(fName + "_wavelet.png", ImageFormat.Png);
            }

            if (routines.Lyapunov != null)
            {
                DataWriter.CreateDataFile(fName + "_lyapunov.txt", routines.Lyapunov.ToString());
            }

            if (chartLyapunov.HasData)
            {
                chartLyapunov.SaveImage(fName + "_lyapunovSlope", ImageFormat.Png);
            }
        }

        private void FillUiWithData()
        {
            fileNameLbl.Text = routines.SourceData.ToString().Replace("\n", " ");

            sourceColumnNum.Maximum = routines.SourceData.ColumnsCount;
            sourceColumnNum.Minimum = 1;

            startPointNum.Maximum = routines.SourceData.Length - 1;

            endPointNum.Maximum = routines.SourceData.Length;
            endPointNum.Value = routines.SourceData.Length;
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
                double tStart = routines.SourceData.TimeSeries.Min.X;
                double tEnd = routines.SourceData.TimeSeries.Max.X;

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

        private void lyapunovRedrawBtn_Click(object sender, EventArgs e)
        {
            try
            {
                if (routines.Lyapunov is KantzMethod)
                {
                    ((KantzMethod)routines.Lyapunov).SetSlope(this.le_kantz_slopeCombo.Text);
                }

                var res = routines.FillLyapunovChart(chartLyapunov, le_pStartNum.ToInt(), le_pEndNum.ToInt(), le_wolf_radio.Checked);

                if (routines.Lyapunov is KantzMethod || routines.Lyapunov is RosensteinMethod)
                {
                    le_resultText.Text = res;
                }
            }
            catch (Exception ex)
            {
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
                    routines.SourceData.TimeSeries.Min.X,
                    routines.SourceData.TimeSeries.Max.X, 
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
            SetLyapunovMethod();

            try
            {
                routines.Lyapunov.Calculate();
                var mi = new MethodInvoker(this.SetLyapunovResult);
                this.BeginInvoke(mi);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Unable to calculate LE:\n" + ex);
                le_resultText.Text = "Error";
            }
        }

        private void SetLyapunovResult()
        {
            le_resultText.BackColor = Color.Khaki;
            string result = string.Empty;

            le_resultText.Text = routines.Lyapunov.GetResult();
            lyap_log_text.Text = routines.Lyapunov.ToString() + "\n\n" + routines.Lyapunov.Log.ToString();

            if (routines.Lyapunov is KantzMethod)
            {
                this.le_kantz_slopeCombo.Items.Clear();
                string[] items = new string[((KantzMethod)routines.Lyapunov).SlopesList.Count];
                ((KantzMethod)routines.Lyapunov).SlopesList.Keys.CopyTo(items, 0);
                this.le_kantz_slopeCombo.Items.AddRange(items);
                this.le_kantz_slopeCombo.SelectedIndex = 0;
                ((KantzMethod)routines.Lyapunov).SetSlope(this.le_kantz_slopeCombo.Text);
            }

            if (routines.Lyapunov.Slope.Length > 1)
            {
                le_pEndNum.Value = routines.Lyapunov.Slope.Length - 1;

                try
                {
                    if (!le_wolf_radio.Checked)
                    {
                        var leSectorEnd = routines.SlopeChangePointIndex(routines.Lyapunov.Slope, 2, routines.Lyapunov.Slope.Amplitude.Y / 30);

                        if (leSectorEnd > 0)
                        {
                            le_pEndNum.Value = leSectorEnd;
                        }
                    }

                    result = routines.FillLyapunovChart(chartLyapunov, le_pStartNum.ToInt(), le_pEndNum.ToInt(), le_wolf_radio.Checked);
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
               
            if (routines.Lyapunov is KantzMethod || routines.Lyapunov is RosensteinMethod)
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
                sourceStepTxt.Text = string.Format(CultureInfo.InvariantCulture, "{0:F8}", routines.SourceData.Step);
            }
            else
            {
                sourceStepTxt.Text = string.Empty;
            }
        }

        private void CleanUp()
        {
            routines.SourceData = null;
            chartSignal.ClearChart();
            chartPoincare.ClearChart();
            chartFft.ClearChart();
            chartLyapunov.ClearChart();
            
            wav_plotPBox.Image = null;
            routines.Lyapunov = null;
            routines.DeleteTempFiles();
        }

        private void mainForm_FormClosing(object sender, FormClosingEventArgs e) =>
            CleanUp();

        private void pointsNum_ValueChanged(object sender, EventArgs e)
        {
            if (this.useTimeCheckbox.Checked)
            {
                RefreshTimeSeries();
                sourceStepTxt.Text = string.Format(CultureInfo.InvariantCulture, "{0:F8}", routines.SourceData.Step);
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
            if (chartSignal.HasData)
            {
                var pf = GetChartPreviewForm(StringData.Signal);
                routines.FillSignalChart(pf.chart);
            }
        }

        private void chartPoincare_DoubleClick(object sender, EventArgs e)
        {
            if (chartPoincare.HasData)
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

        private void SetLyapunovMethod()
        {
            int dim = le_dimNum.ToInt();
            int tau = le_tauNum.ToInt();
            double scaleMin = le_scaleMinNum.ToDouble();

            if (le_wolf_radio.Checked)
            {
                routines.Lyapunov = new WolfMethod(
                    routines.SourceData.TimeSeries.YValues,
                    dim,
                    tau,
                    le_wolf_stepNum.ToDouble(),
                    scaleMin,
                    le_scaleMaxNum.ToDouble(),
                    le_wolf_evolveStepsNum.ToInt()
                );
            }
            else if (lyap_calc_Rad_rosenstein.Checked)
            {
                routines.Lyapunov = new RosensteinMethod(
                    routines.SourceData.TimeSeries.YValues,
                    dim,
                    tau,
                    le_ros_stepsNum.ToInt(),
                    le_ros_distanceNum.ToInt(),
                    scaleMin
                );
            }
            else if (lyap_calc_Rad_kantz.Checked)
            {
                routines.Lyapunov = new KantzMethod(
                    routines.SourceData.TimeSeries.YValues,
                    dim,
                    tau,
                    le_ros_stepsNum.ToInt(),
                    le_ros_distanceNum.ToInt(),
                    scaleMin,
                    le_scaleMaxNum.ToDouble(),
                    le_kantz_scalesNum.ToInt()
                );
            }
            else if (le_jakobian_radio.Checked)
            {
                routines.Lyapunov = new JakobianMethod(
                    routines.SourceData.TimeSeries.YValues,
                    dim,
                    routines.SourceData.TimeSeries.YValues.Length,
                    scaleMin,
                    1.2,
                    30,
                    false
                );
            }
        }

        private void lyap_calc_Rad_kantz_CheckedChanged(object sender, EventArgs e)
        {
            this.le_kantz_slopeCombo.Visible = (sender as RadioButton).Checked;
        }
    }
}
