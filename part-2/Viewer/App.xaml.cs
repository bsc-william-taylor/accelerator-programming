using System.Windows;
using System.IO;
using System.Drawing;
using System.ComponentModel;
using System;
using System.Text;

namespace Viewer
{
    public partial class App : Application
    {
        public static string GetByte(char[] bytes, ref int pointer, StringBuilder builder)
        {   
            while (true)
            {
                char temp = bytes[++pointer];

                while (true)
                {
                    if ((temp == '\n' || temp == '\r') && builder.Length == 0)
                        temp = bytes[++pointer];
                    else
                        break;
                }

                if (temp != ' ' && temp != '\n' && temp != '\r')
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

            using (var reader = new BinaryReader(new FileStream(file, FileMode.Open)))
            {
                var bufferPosition = -1;
                var builder = new StringBuilder();
                var bytes = reader.ReadChars((int)reader.BaseStream.Length);
                var magic = GetByte(bytes, ref bufferPosition, builder);
                var width = int.Parse(GetByte(bytes, ref bufferPosition, builder));
                var height = int.Parse(GetByte(bytes, ref bufferPosition, builder));
                var maxValue = int.Parse(GetByte(bytes, ref bufferPosition, builder));

                Bitmap bitmap = new Bitmap(width, height);

                for (int y = 0; y < height; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        try 
                        {
                            var r = int.Parse(GetByte(bytes, ref bufferPosition, builder));
                            var g = int.Parse(GetByte(bytes, ref bufferPosition, builder));
                            var b = int.Parse(GetByte(bytes, ref bufferPosition, builder));

                            bitmap.SetPixel(x, y, Color.FromArgb(r, g, b));
                        }
                        catch(Exception e)
                        {
                            MessageBox.Show($"Error reading at pixel -> {x}:{y}");
                        }
                    }

                    var progress = y / (double)height * 100.0;
                    worker.ReportProgress((int)progress);
                }

                return bitmap;
            }
        }
    }
}
