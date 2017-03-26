﻿using System.Windows;
using System.Windows.Media.Imaging;
using System.IO;
using System.Windows.Forms;
using System.ComponentModel;
using System.Collections.Generic;
using System;
using System.Text.RegularExpressions;
using InputKey = System.Windows.Input.Key;
using Forms = System.Windows.Forms;

namespace Viewer
{
    public partial class MainWindow : Window
    {
        List<BitmapImage> LoadedImages = new List<BitmapImage>();
        int LoadedImageIndex = 0;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void ViewImage(object sender, RoutedEventArgs e)
        {
            var fileDialog = new OpenFileDialog();
            fileDialog.Multiselect = false;
            fileDialog.Filter = "PPM|*.ppm";

            if (fileDialog.ShowDialog() == Forms.DialogResult.OK)
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
            LoadedImages.Add((BitmapImage)e.Result);
            LoadedImageIndex++;
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

        private void About(object sender, RoutedEventArgs e)
        {
            var title = "Information";
            var body = Regex.Replace(@"
                This is a viewer for PPM files. Use right and left arrows to cycle 
                through loaded PPM files and press the clear menu item to unload all images.
            ", @"\s+", " ");

            Forms.MessageBox.Show(body, title, MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void Window_KeyDown(object sender, System.Windows.Input.KeyEventArgs e)
        {
            if (e.Key == InputKey.Right && LoadedImageIndex + 1 < LoadedImages.Count)
            {
                ImageView.Source = LoadedImages[++LoadedImageIndex];
            }

            if (e.Key == InputKey.Left && LoadedImageIndex - 1 >= 0)
            {
                ImageView.Source = LoadedImages[--LoadedImageIndex];
            }
        }

        private void ClearCache(object sender, RoutedEventArgs e)
        {
            LoadedImages.Clear();
            LoadedImageIndex = 0;

            BitmapImage bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.UriSource = new Uri("http://placehold.it/600x620", UriKind.RelativeOrAbsolute);
            bitmap.EndInit();

            ImageView.Source = bitmap;
        }
    }
}
