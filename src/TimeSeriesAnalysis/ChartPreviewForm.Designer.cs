using System.Windows.Forms;
using MathLib.DrawEngine;

namespace TimeSeriesAnalysis
{
    partial class ChartPreviewForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ChartPreviewForm));
            this.contextMenu = new ContextMenu();
            this.saveAsItem = new MenuItem();
            this.copyItem = new MenuItem();
            this.chart = new MathChart();
            ((System.ComponentModel.ISupportInitialize)(this.chart)).BeginInit();
            this.SuspendLayout();
            // 
            // contextMenu
            // 
            this.contextMenu.MenuItems.AddRange(new MenuItem[] {
            this.saveAsItem,
            this.copyItem});
            // 
            // saveAsItem
            // 
            this.saveAsItem.Index = 0;
            this.saveAsItem.Text = "Save as";
            this.saveAsItem.Click += new System.EventHandler(this.ContextMenuClick);
            // 
            // copyItem
            // 
            this.copyItem.Index = 1;
            this.copyItem.Text = "Copy";
            this.copyItem.Click += new System.EventHandler(this.copyItem_Click);
            // 
            // chart
            // 
            this.chart.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
            this.chart.Location = new System.Drawing.Point(1, 1);
            this.chart.Name = "chart";
            this.chart.Size = new System.Drawing.Size(1007, 559);
            this.chart.TabIndex = 13;
            this.chart.Text = "chart1";
            this.chart.MouseClick += new System.Windows.Forms.MouseEventHandler(this.chart_MouseClick);
            // 
            // ChartPreviewForm
            // 
            this.ClientSize = new System.Drawing.Size(1008, 561);
            this.Controls.Add(this.chart);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MinimumSize = new System.Drawing.Size(133, 100);
            this.Name = "ChartPreviewForm";
            this.Text = "Preview Form";
            ((System.ComponentModel.ISupportInitialize)(this.chart)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion
        public ContextMenu contextMenu;
        private MenuItem saveAsItem;
        private MenuItem copyItem;
        public MathChart chart;
    }
}