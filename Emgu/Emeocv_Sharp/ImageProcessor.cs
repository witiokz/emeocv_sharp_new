using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using EmeocvSharp;
using Emgu.CV;
using System.Drawing;
using Emgu.CV.Util;
using Emgu.CV.Structure;
using System.IO;
using Emgu.CV.CvEnum;

namespace Emeocv_Sharp
{
    public class ImageProcessor : IImageProcessor, IDisposable
    {
        private Config _config { get; set; }
        private Image<Bgr, byte> _img { get; set; }
        private Image<Gray, byte> _imgGray { get; set; }
        private List<Mat> _digits { get; set; }
        private bool _debugWindow { get; set; }
        private bool _debugSkew { get; set; }
        private bool _debugEdges { get; set; }
        private bool _debugDigits { get; set; }

        public ImageProcessor()
        {
            _config = new Config();
            DebugWindow(false);
            DebugSkew(false);
            DebugDigits(false);
            DebugEdges(false);
            _digits = new List<Mat>();
        }
        public void DebugDigits(bool bval = true)
        {
            _debugDigits = bval;
        }

        public void DebugEdges(bool bval = true)
        {
            _debugEdges = bval;
        }

        public void DebugSkew(bool bval = true)
        {
            _debugSkew = bval;
        }

        public void DebugWindow(bool bval = true)
        {
            _debugWindow = bval;

            if (_debugWindow)
            {
                CvInvoke.NamedWindow("ImageProcessor");
            }
        }

        //Set the input image.
        public void SetInput(Image<Bgr, byte> img)
        {
            _img = img;

            //_imgGray = img.Convert<Gray, byte>();
        }

        //Get the vector of output images.
        //Each image contains the edges of one isolated digit.
        public List<Mat> GetOutput()
        {
            return _digits;
        }

        public void ShowImage()
        {
            CvInvoke.Imshow("ImageProcessor", _img);
        }

 
        //Main processing function.
        //Read input image and create vector of images for each digit.
        public void Process()
        {
            _digits.Clear();

            // convert to gray
            _imgGray = _img.Convert<Gray, byte>();

            // initial rotation to get the digits up
            Rotate(_config.rotationDegrees);

            // detect and correct remaining skew (+- 30 deg)
            float skew_deg = DetectSkew();
            Rotate(skew_deg);


            // find and isolate counter digits
            FindCounterDigits();

            if (_debugWindow)
            {
                //ShowImage();
            }
        }

        //Rotate image.
        private void Rotate(double rotationDegrees)
        {
            Mat mapMatrix = new Mat();
            Image<Gray, byte> img_rotated = _imgGray;
            CvInvoke.GetRotationMatrix2D(new PointF(this._imgGray.Cols / 2, this._imgGray.Rows / 2), rotationDegrees, 1, mapMatrix);

            CvInvoke.WarpAffine(_imgGray, img_rotated, mapMatrix, _img.Size);
            _imgGray = img_rotated;

            if (_debugWindow)
            {
                //CvInvoke.WarpAffine(_img, img_rotated, mapMatrix, _img.Size);
                //TODO:
                //_img = img_rotated;
            }

            //TODO:
            RotateDebug(rotationDegrees);
        }

        private void RotateDebug(double rotationDegrees)
        {
            UMat mapMatrix = new UMat();
            Image<Bgr, byte> img_rotated = this._img;
            CvInvoke.GetRotationMatrix2D(new PointF(this._img.Cols / 2, this._img.Rows / 2), rotationDegrees, 1, mapMatrix);

            CvInvoke.WarpAffine(this._img, img_rotated, mapMatrix, this._img.Size);
            _img = img_rotated;
        }

        //Draw lines into image.
        //For debugging purposes.
        private void DrawLines(VectorOfPointF lines)
        {
            // draw lines
            for (var i = 0; i < lines.Size; i++)
            {
                float rho = lines[i].X;
                float theta = lines[i].Y;
                double a = Math.Cos(theta), b = Math.Sin(theta);
                double x0 = a * rho, y0 = b * rho;
                var pt1 = new Point((int)Math.Round(x0 + 1000 * (-b)), (int)Math.Round(y0 + 1000 * (a)));
                var pt2 = new Point((int)Math.Round(x0 - 1000 * (-b)), (int)Math.Round(y0 - 1000 * (a)));

                CvInvoke.Line(_img, pt1, pt2, new MCvScalar(255, 0, 0), 1);
            }
        }

        //Draw lines into image.
        //For debugging purposes.
        private void DrawLines(VectorOfPointF lines, int xoff = 0, int yoff = 0)
        {
            for (var i = 0; i < lines.Size; i++)
            {
                CvInvoke.Line(_img, new Point((int)(lines[i].X + xoff), (int)(lines[i].Y + yoff)),
                new Point((int)(lines[i].X + xoff), (int)(lines[i].Y + yoff)), new MCvScalar(255, 0, 0), 1);
            }
        }

        //Detect the skew of the image by finding almost (+- 30 deg) horizontal lines.
        private float DetectSkew()
        {
            Mat edges = CannyEdges();

            // find lines
            VectorOfPointF lines = new VectorOfPointF();

            // filter lines by theta and compute average
            List<PointF> filteredLines = new List<PointF>();

            CvInvoke.HoughLines(edges, lines, 1, Math.PI / 180, 140);

            float theta_min = (float)(60 * Math.PI / 180);
            float theta_max = (float)(120 * Math.PI / 180);
            float theta_avr = 0;
            float theta_deg = 0;
            for (var i = 0; i < lines.Size; i++)
            {
                float theta = lines[i].Y;
                if (theta >= theta_min && theta <= theta_max)
                {
                    filteredLines.Add(lines[i]);
                    theta_avr += theta;
                }
            }

            if (filteredLines.Count > 0)
            {
                theta_avr /= filteredLines.Count;
                theta_deg = (float)(theta_avr / Math.PI * 180) - 90;
            }

            if (_debugSkew)
            {
                //DrawLines(filteredLines);
            }

            return theta_deg;
        }

        //Detect edges using Canny algorithm.
        private Mat CannyEdges()
        {
            Mat edges = new Mat();
            CvInvoke.Canny(_imgGray, edges, _config.cannyThreshold1, _config.cannyThreshold2);

            return edges;
        }

        //Find bounding boxes that are aligned at y position.
        private void FindAlignedBoxes(List<Rectangle> list, int begin, int end, List<Rectangle> result)
        {
            var start = list[begin];
            result.Add(start);

            begin = begin + 1;
            for (var index = begin; index < end; index++)
            {
                if (Math.Abs(start.Y - list[index].Y) < _config.digitYAlignment && Math.Abs(start.Height - list[index].Height) < 5)
                {
                    result.Add(list[index]);
                }
            }
        }

        //Filter contours by size of bounding rectangle.
        private void FilterContours(VectorOfVectorOfPoint contours, List<Rectangle> boundingBoxes, VectorOfVectorOfPoint filteredContours)
        {
            // filter contours by bounding rect size
            for (int i = 0; i < contours.Size; i++)
            {
                var bounds = CvInvoke.BoundingRectangle(contours[i]);

                //TODO:
                if (bounds.Height > _config.digitMinHeight && bounds.Height < _config.digitMaxHeight
                        && bounds.Width > 5 && bounds.Width < bounds.Height)
                {
                //if (true)
                //{
                    boundingBoxes.Add(bounds);
                    filteredContours.Push(contours[i]);

                    
                }
            }
        }

        //Find and isolate the digits of the counter,
        private void FindCounterDigits()
        {
            var runningId = (new Random()).Next(100000);
            Mat edges = CannyEdges();
            if (this._debugEdges)
            {
                //CvInvoke.Imshow("edges", edges);
            }

            Mat img_ret = edges.Clone();

            //find contours in whole image
            List<Rectangle> boundingBoxes = new List<Rectangle>();
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            VectorOfVectorOfPoint filteredContours = new VectorOfVectorOfPoint();

            //Find contours
            CvInvoke.FindContours(edges, contours, null, RetrType.External, ChainApproxMethod.ChainApproxNone);

            // filter contours by bounding rect size
            FilterContours(contours, boundingBoxes, filteredContours);

            //Draw contours
            var backedUp = _img.Clone();
            List<Rectangle> bounds = new List<Rectangle>();
            for (var i = 0; i < contours.Size; i++)
            {
                bounds.Add(CvInvoke.BoundingRectangle(contours[i]));
            }

            //Filter contourn // sort bounding boxes from left to right
            bounds = bounds.Where(c => c.Width > 10 && c.Height > 10).OrderBy(c => c.Top).ThenBy(c => c.Left).ToList();

            var count = 0;
            foreach (var bound in bounds)
            {
                count++;
                _img.Draw(bound, new Bgr(Color.Green), 1);
                var contourImage = backedUp.Clone();
                contourImage.ROI = bound;

                _digits.Add(img_ret);

                var contourImageName = string.Format(Path.Combine(Config.PathToOutput, "{0}_{1}.jpg"), runningId, count);
                contourImage.Save(contourImageName);
                Console.WriteLine(contourImageName);

            }

            _img.Save(Path.Combine(Config.PathToOutput, "contouredImage.jpg"));
        }

        //Find and isolate the digits of the counter,
        private void FindCounterDigits_()
        {
            Mat edges = CannyEdges();
            if (this._debugEdges)
            {
                CvInvoke.Imshow("edges", edges);
            }

            Mat img_ret = edges.Clone();

            //find contours in whole image
            List<Rectangle> boundingBoxes = new List<Rectangle>();
            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            VectorOfVectorOfPoint filteredContours = new VectorOfVectorOfPoint();

            //Find contours
            CvInvoke.FindContours(edges, contours, null, RetrType.External, ChainApproxMethod.ChainApproxNone);

            // filter contours by bounding rect size
            FilterContours(contours, boundingBoxes, filteredContours);

            //find bounding boxes that are aligned at y position
            var alignedBoundingBoxes = new List<Rectangle>();
            var tmpRes = new List<Rectangle>();
            for (var index = 0; index < boundingBoxes.Count; index++)
            {
                tmpRes.Clear();
                FindAlignedBoxes(boundingBoxes, index, boundingBoxes.Count, tmpRes);
                if (tmpRes.Count > alignedBoundingBoxes.Count)
                {
                    alignedBoundingBoxes = tmpRes;
                }
            }

            // sort bounding boxes from left to right
            alignedBoundingBoxes = alignedBoundingBoxes.Where(c => c.Width > 10 && c.Height > 10).OrderBy(c => c.Top).ThenBy(c => c.Left).ToList();

            if (_debugEdges)
            {
                // draw contours
                Matrix<int> cont = new Matrix<int>(edges.Rows, edges.Cols, 0);
                cont.SetZero();

                CvInvoke.DrawContours(cont, filteredContours, -1, new MCvScalar(255));
                CvInvoke.Imshow("contours", cont);
            }

            // cut out found rectangles from edged image
            for (int i = 0; i < alignedBoundingBoxes.Count; ++i)
            {
                Rectangle roi = alignedBoundingBoxes[i];
                _digits.Add(img_ret);

                if (_debugDigits)
                {
                    CvInvoke.Rectangle(_img, roi, new MCvScalar(0, 255, 0), 2);
                }

                var img =_imgGray.Clone();
                img.ROI = roi;
                img.Save(string.Format(Path.Combine(Config.PathToOutput, "{0}_{1}.jpg"), i, Guid.NewGuid()));
               
            }
        }

        public void Dispose()
        {
        }
    }
}
