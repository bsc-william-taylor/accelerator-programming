using System.Windows;
using System.Windows.Media.Imaging;
using System.IO;
using System.Windows.Forms;
using Forms = System.Windows.Forms;
using System.ComponentModel;

namespace Viewer
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void ViewImage(object sender, RoutedEventArgs e)
        {
            var fileDialog = new OpenFileDialog();
            fileDialog.Multiselect = false;
            
            if(fileDialog.ShowDialog() == Forms.DialogResult.OK)
            {
                BackgroundWorker worker = new BackgroundWorker();
                worker.WorkerReportsProgress = true;
                worker.DoWork += LoadImage;
                worker.ProgressChanged += Worker_ProgressChanged;
                worker.RunWorkerCompleted += Worker_RunWorkerCompleted;
                worker.RunWorkerAsync(fileDialog.FileName);
            }
        }

        private void Worker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            ImageView.Source = (BitmapImage)e.Result;
        }

        private void Worker_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            Progress.Value = e.ProgressPercentage;
        }

        private void LoadImage(object sender, DoWorkEventArgs e)
        {
            var worker = sender as BackgroundWorker;
            var filename = e.Argument.ToString();
            var bitmap = App.ReadBitmapFromPPM(filename, sender as BackgroundWorker);

            using (MemoryStream memory = new MemoryStream())
            {
                bitmap.Save(memory, System.Drawing.Imaging.ImageFormat.Bmp);
                memory.Position = 0;

                BitmapImage bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = memory;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
                bitmapImage.Freeze();

                e.Result = bitmapImage;

                worker.ReportProgress(100);
            }
        }

        private void Run(object sender, RoutedEventArgs e)
        {
            Forms.MessageBox.Show("Run");
        }

        private void Settings(object sender, RoutedEventArgs e)
        {
            Forms.MessageBox.Show("Settings");
        }

        private void About(object sender, RoutedEventArgs e)
        {
            Forms.MessageBox.Show("About");
        }
    }
}
