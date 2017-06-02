using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Emgu.CV;
using EmeocvSharp;
using Emgu.CV.ML;
using System.IO;
using System.Text.RegularExpressions;
using System.Runtime.InteropServices;

namespace Emeocv_Sharp
{
    //OCR to train and recognize digits with the KNearest model.
    public class KNearestOcr
    {
        Mat _samples = new Mat();
        Mat _responses = new Mat();

        KNearest _pModel;
        Config _config = new Config();

        //Learn a single digit.
        int Learn(Mat img)
        {
            CvInvoke.Imshow("Learn", img);
            var key = CvInvoke.WaitKey(0);

            if (key >= '0' && key <= '9')
            {
                _responses.PushBack(new Mat(1, 1, Emgu.CV.CvEnum.DepthType.Cv32F, key - '0'));
                _samples.PushBack(PrepareSample(img));
            }

            return key;
        }

        //Learn a vector of digits.
        public int Learn(List<Mat> images)
        {
            var key = 0;

            foreach (var it in images)
            {
                if (key != 's' && key != 'q')
                {
                    key = Learn(it);
                }

            }

            return key;
        }

        //Save training data to file.
        public void SaveTrainingData()
        {
            var str = new StringBuilder();

            str.Append("%YAML:1.0" + Environment.NewLine);

            //samples
            str.Append("samples: !!opencv-matrix" + Environment.NewLine);
            str.Append("rows: " + _samples.Rows + Environment.NewLine);
            str.Append("cols: " + _samples.Cols + Environment.NewLine);
            str.Append("dt: " + "f" + Environment.NewLine);
            str.Append("data: " + _samples.Data + Environment.NewLine);

            //responses
            str.Append("responses: !!opencv-matrix" + Environment.NewLine);
            str.Append("rows: " + _responses.Rows + Environment.NewLine);
            str.Append("cols: " + _responses.Cols + Environment.NewLine);
            str.Append("dt: " + "f" + Environment.NewLine);
            str.Append("data: " + _responses.Data);

            File.WriteAllText(_config.trainingDataFilename, str.ToString());
        }

        //Load training data from file and init model.
        public bool LoadTrainingData()
        {
            var content = File.ReadAllText(_config.trainingDataFilename);

            var pattern = "(.*):(.*?)(\n|$)";

            var matches = Regex.Matches(content, pattern);

            if (matches.Count > 0)
            {
                for (int i = 0; i < matches.Count; i++)
                {
                    var item = matches[i];

                    if (new[] { "samples", "responses" }.Contains(item.Groups[1].Value))
                    {
                        var rows = int.Parse(matches[i + 1].Value.Split(':')[1].Trim());
                        var cols = int.Parse(matches[i + 2].Value.Split(':')[1].Trim());
                        var dt = matches[i + 3].Value.Split(':')[1].Trim() == "f" ? Emgu.CV.CvEnum.DepthType.Cv64F : Emgu.CV.CvEnum.DepthType.Default;
                        var data = Marshal.UnsafeAddrOfPinnedArrayElement(matches[i + 3].Value.Split(':')[1].Trim().Replace("[", "").Replace("]", "").Replace(".", ".0").Trim().Split(','), 0);
                        var mat = new Mat(new[] { rows, cols }, dt, data);

                        if (item.Groups[1].Value == "samples")
                        {
                            _samples = mat;
                        }

                        if (item.Groups[1].Value == "responses")
                        {
                            _responses = mat;
                        }
                    }
                }
            }

            InitModel();

            return true;
        }

        //Recognize a single digit.
        public int Recognize(Mat img)
        {
            int cres = '?';

            Mat results = new Mat();
            Matrix<float> neighborResponses = new Matrix<float>(_samples.Rows, 10);
            int dists = 0;

            float result = _pModel.Predict(PrepareSample(img), neighborResponses);

            if (0 == neighborResponses.Data[0,0] - neighborResponses.Data[0, 1])
            {
                //&& dists.at<float>(0, 0) < _config.getOcrMaxDist()
                // valid character if both neighbors have the same value and distance is below ocrMaxDist
                cres = '0' + (int)result;
            }

            Console.WriteLine("results: " + results);
            Console.WriteLine("neighborResponses: " + neighborResponses);
            Console.WriteLine("dists: " + dists);
            Console.WriteLine("results: " + results);
            Console.WriteLine("results: " + results);

            return cres;
        }

        //Recognize a vector of digits.
        public string Recognize(List<Mat> images)
        {
            var result = string.Empty;

            foreach (var it in images)
            {
                result += Recognize(it);
            }
            return result;
        }

        //Prepare an image of a digit to work as a sample for the model.
        Mat PrepareSample(Mat img)
        {
            Mat roi = new Mat();
            Mat sample = new Mat();
            CvInvoke.Resize(img, roi, new System.Drawing.Size(10, 10));
            roi.Reshape(1, 1).ConvertTo(sample, Emgu.CV.CvEnum.DepthType.Cv32F);
            return sample;
        }

        //Initialize the model.
        void InitModel()
        {
            _pModel = new KNearest(); //new KNearest(_samples, _responses);
        }
    }
}

