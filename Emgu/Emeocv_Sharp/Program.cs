using EmeocvSharp;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Emeocv_Sharp
{
    class Program
    {
        //https://www.mkompf.com/cplus/emeocv.html

        static int delay = 1000;
        //#define VERSION "0.9.6"

        static void Main(string[] args)
        {
            //TestOcr(@"C:\Users\user\Desktop\test");
            //return;

           using (ImageProcessor processor = new ImageProcessor())
            {
                processor.SetInput(new Image<Bgr, byte>(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, @"img\00.png")));
                processor.Process();
                var t = processor.GetOutput();
            }
        }

        static void TestOcr(string pImageInputPath)
        {
            ImageProcessor proc = new ImageProcessor();
            proc.DebugWindow();
            proc.DebugDigits();

            Plausi plausi = new Plausi();

            KNearestOcr ocr = new KNearestOcr();
            if (!ocr.LoadTrainingData())
            {
                Console.WriteLine("Failed to load OCR training data\n");
                return;
            }
            Console.WriteLine("OCR training data loaded.\n");
            Console.WriteLine("<q> to quit.\n");

            var images = Directory.GetFiles(pImageInputPath, "*.png");

            foreach (var image in images)
            {
                proc.SetInput(new Image<Bgr, byte>(image));
                proc.Process();

                var output = proc.GetOutput();

                var result = ocr.Recognize(output);
                Console.WriteLine("Result: " + result);

                if (plausi.Check(result, DateTime.Now))
                {
                    Console.WriteLine(plausi.CheckedValue);
                }
                else
                {
                    Console.WriteLine("  -------");
                }
                int key = CvInvoke.WaitKey(delay);

                if (key == 'q')
                {
                    Console.WriteLine("Quit\n");
                    break;
                }
            }
        }

        static void LearnOcr(string pImageInputPath)
        {
            ImageProcessor proc = new ImageProcessor();
            proc.DebugWindow();

            KNearestOcr ocr = new KNearestOcr();
            ocr.LoadTrainingData();
            Console.WriteLine("Entering OCR training mode!\n");
            Console.WriteLine("<0>..<9> to answer digit, <space> to ignore digit, <s> to save and quit, <q> to quit without saving.\n");

            var images = Directory.GetFiles(pImageInputPath);
            int key = 0;

            foreach (var image in images)
            {
                proc.SetInput(new Image<Bgr, byte>(image));
                proc.Process();

                key = ocr.Learn(proc.GetOutput());
                
                if (key == 'q' || key == 's')
                {
                    Console.WriteLine("Quit\n");
                    break;
                }
            }

            if (key != 'q')
            {
                Console.WriteLine("Saving training data\n");
                ocr.SaveTrainingData();
            }
        }

    }

    public class ContourWithData
    {
        // member variables
        const int MIN_CONTOUR_AREA = 100;
        //contour
        public VectorOfPoint contour;
        //bounding rect for contour
        public Rectangle boundingRect;
        //area of contour
        public double dblArea;

        public bool CheckIfContourIsValid()
        {
            //this is oversimplified, for a production grade program better validity checking would be necessary
            if ((dblArea < MIN_CONTOUR_AREA))
            {
                return false;
            }
            else
            {
                return true;
            }
        }
    }
}
