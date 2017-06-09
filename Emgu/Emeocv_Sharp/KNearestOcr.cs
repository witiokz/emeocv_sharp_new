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
using Emgu.CV.CvEnum;
using System.Drawing;
using Emgu.CV.Util;
using Emgu.CV.Structure;

namespace Emeocv_Sharp
{
    //OCR to train and recognize digits with the KNearest model.
    public class KNearestOcr
    {
        Matrix<float> _samples = null;
        Matrix<float> _responses = null;

        KNearest _pModel;
        Config _config = new Config();

        //Learn a single digit.
        int Learn(Mat img)
        {
            CvInvoke.Imshow("Learn", img);
            var key = CvInvoke.WaitKey(0);

            if (key >= '0' && key <= '9')
            {
                _responses.Add(key - '0');
                _samples.Add(PrepareSample(img));
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
                        var data = matches[i + 4].Value.Split(':')[1].Trim().Replace("[", "").Replace("]", "").Replace(".", ".0").Trim().Split(',').Select(j => float.Parse(j)).ToArray();

                        var mat = new Matrix<float>(data);

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
            const int RESIZED_IMAGE_WIDTH = 10;
            const int RESIZED_IMAGE_HEIGHT = 10;

            int cres = '?';

            var mtxClassifications = _responses;
            int intNumberOfTrainingSamples = mtxClassifications.Rows;

            mtxClassifications = new Matrix<float>(447, 1);
            var mtxTrainingImages = new Matrix<float>(447, 100);

            //TODO:
              mtxTrainingImages = _samples;

            // train 
            KNearest kNearest = new KNearest();
            kNearest.DefaultK = 1;
            kNearest.Train(mtxTrainingImages, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample, mtxClassifications);

            Mat imgTestingNumbers = img;

            //declare various images
            Mat imgGrayscale = new Mat();
            Mat imgBlurred = new Mat();
            Mat imgThresh = new Mat();
            Mat imgThreshCopy = new Mat();

            //convert to grayscale
            CvInvoke.CvtColor(imgTestingNumbers, imgGrayscale, ColorConversion.Bgr2Gray);

            //blur
            CvInvoke.GaussianBlur(imgGrayscale, imgBlurred, new Size(5, 5), 0);

            //threshold image from grayscale to black and white
            CvInvoke.AdaptiveThreshold(imgBlurred, imgThresh, 255.0, AdaptiveThresholdType.GaussianC, ThresholdType.BinaryInv, 11, 2.0);

            //make a copy of the thresh image, this in necessary b/c findContours modifies the image
            imgThreshCopy = imgThresh.Clone();

            var contours = new VectorOfVectorOfPoint();

            //get external countours only
            CvInvoke.FindContours(imgThreshCopy, contours, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

            //declare a list of contours with data
            var listOfContoursWithData = new List<ContourWithData>();

            //populate list of contours with data
            //for each contour
            for (int i = 0; i <= contours.Size - 1; i++)
            {
                //declare new contour with data
                ContourWithData contourWithData = new ContourWithData();
                //populate contour member variable
                contourWithData.contour = contours[i];
                //calculate bounding rectangle
                contourWithData.boundingRect = CvInvoke.BoundingRectangle(contourWithData.contour);
                //calculate area
                contourWithData.dblArea = CvInvoke.ContourArea(contourWithData.contour);

                //if contour with data is valid
                if ((contourWithData.CheckIfContourIsValid()))
                {
                    //add to list of contours with data
                    listOfContoursWithData.Add(contourWithData);
                }
            }
            //sort contours with data from left to right
            listOfContoursWithData.Sort((oneContourWithData, otherContourWithData) => oneContourWithData.boundingRect.X.CompareTo(otherContourWithData.boundingRect.X));

            //declare final string, this will have the final number sequence by the end of the program
            string strFinalString = "";

            //for each contour in list of valid contours
            foreach (ContourWithData contourWithData in listOfContoursWithData)
            {
                //draw green rect around the current char
                CvInvoke.Rectangle(imgTestingNumbers, contourWithData.boundingRect, new MCvScalar(0.0, 255.0, 0.0), 2);

                //get ROI image of bounding rect
                Mat imgROItoBeCloned = new Mat(imgThresh, contourWithData.boundingRect);

                //clone ROI image so we don't change original when we resize
                Mat imgROI = imgROItoBeCloned.Clone();

                Mat imgROIResized = new Mat();

                //resize image, this is necessary for char recognition
                CvInvoke.Resize(imgROI, imgROIResized, new Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

                //declare a Matrix of the same dimensions as the Image we are adding to the data structure of training images
                Matrix<float> mtxTemp = new Matrix<float>(imgROIResized.Size);

                //declare a flattened (only 1 row) matrix of the same total size
                Matrix<float> mtxTempReshaped = new Matrix<float>(1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT);

                //convert Image to a Matrix of Singles with the same dimensions
                imgROIResized.ConvertTo(mtxTemp, DepthType.Cv32F);

                //flatten Matrix into one row by RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT number of columns
                for (int intRow = 0; intRow <= RESIZED_IMAGE_HEIGHT - 1; intRow++)
                {
                    for (int intCol = 0; intCol <= RESIZED_IMAGE_WIDTH - 1; intCol++)
                    {
                        mtxTempReshaped[0, (intRow * RESIZED_IMAGE_WIDTH) + intCol] = mtxTemp[intRow, intCol];
                    }
                }

                float sngCurrentChar = 0;

                //finally we can call Predict !!!
                sngCurrentChar = kNearest.Predict(mtxTempReshaped);

                //append current char to full string of chars 
                strFinalString = strFinalString + (char)sngCurrentChar;
            }

            //Console.WriteLine("results: " + results);
            //Console.WriteLine("neighborResponses: " + neighborResponses);
            //Console.WriteLine("dists: " + dists);
            //Console.WriteLine("results: " + results);

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
        Matrix<float> PrepareSample(Mat img)
        {
            Mat roi = new Mat();
            Matrix<float> sample = new Matrix<float>(10,10);
            CvInvoke.Resize(img, roi, new Size(10, 10));
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

