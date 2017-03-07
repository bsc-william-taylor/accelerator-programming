using System.Windows;
using System.IO;
using System.Drawing;
using System.ComponentModel;
using System;
using System.Text;
using System.Windows.Threading;
using System.Threading.Tasks;
namespace Viewer
{
    public partial class App : Application
    {
        public static string GetByte(BinaryReader reader, StringBuilder builder)
        {
            while (reader.BaseStream.Position != reader.BaseStream.Length)
            {
                char temp = reader.ReadChar();

                while(reader.BaseStream.Position != reader.BaseStream.Length)
                {
                    if (temp == '\n' && builder.Length == 0)
                        temp = reader.ReadChar();
                    else
                        break;
                }

                if (temp != ' ' && temp != '\n')
                    builder.Append(temp);
                else
                    break;
            }

            var str = builder.ToString();
            builder.Clear();
            return str;
        }

        public static Bitmap ReadBitmapFromPPM(string file, BackgroundWorker worker)
        {
            worker.ReportProgress(0);

            var builder = new StringBuilder();
            var reader = new BinaryReader(new FileStream(file, FileMode.Open));
            var magic = GetByte(reader, builder);
            var width = int.Parse(GetByte(reader, builder));
            var height = int.Parse(GetByte(reader, builder));
            var maxValue = int.Parse(GetByte(reader, builder));
            var totalWork = width * height;

            Bitmap bitmap = new Bitmap(width, height);

            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    var progress = (double)(width * y + x) / (double)totalWork;
                    var r = int.Parse(GetByte(reader, builder));
                    var g = int.Parse(GetByte(reader, builder));
                    var b = int.Parse(GetByte(reader, builder));

                    bitmap.SetPixel(x, y, Color.FromArgb(r, g, b));
                    worker.ReportProgress((int)(progress * 100.0));
                }                
            }

            return bitmap;
        }

        private void Application_Startup(object sender, StartupEventArgs e)
        {
            var processors = Environment.ProcessorCount;

            MessageBox.Show(processors.ToString());
        }
    }
}
