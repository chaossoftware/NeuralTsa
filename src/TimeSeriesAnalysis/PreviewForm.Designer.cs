using System;
using System.Windows.Forms;

namespace TimeSeriesAnalysis
{
    partial class PreviewForm
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(PreviewForm));
            this.contextMenu = new System.Windows.Forms.ContextMenu();
            this.saveAsItem = new System.Windows.Forms.MenuItem();
            this.copyItem = new System.Windows.Forms.MenuItem();
            this.previewPBox = new System.Windows.Forms.PictureBox();
            ((System.ComponentModel.ISupportInitialize)(this.previewPBox)).BeginInit();
            this.SuspendLayout();
            // 
            // contextMenu
            // 
            this.contextMenu.MenuItems.AddRange(new System.Windows.Forms.MenuItem[] {
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
            // previewPBox
            // 
            this.previewPBox.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.previewPBox.BackColor = System.Drawing.Color.White;
            this.previewPBox.Cursor = System.Windows.Forms.Cursors.Cross;
            this.previewPBox.Location = new System.Drawing.Point(0, 0);
            this.previewPBox.Name = "previewPBox";
            this.previewPBox.Size = new System.Drawing.Size(1008, 561);
            this.previewPBox.TabIndex = 0;
            this.previewPBox.TabStop = false;
            this.previewPBox.SizeChanged += new System.EventHandler(this.previewPBox_SizeChanged);
            this.previewPBox.MouseClick += new System.Windows.Forms.MouseEventHandler(this.PreviewContextMenu);
            // 
            // PreviewForm
            // 
            this.ClientSize = new System.Drawing.Size(1008, 561);
            this.Controls.Add(this.previewPBox);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.SizableToolWindow;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MinimumSize = new System.Drawing.Size(133, 100);
            this.Name = "PreviewForm";
            this.Text = "Preview Form";
            ((System.ComponentModel.ISupportInitialize)(this.previewPBox)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion
        public ContextMenu contextMenu;
        private MenuItem saveAsItem;
        private MenuItem copyItem;
        public PictureBox previewPBox;
    }
}