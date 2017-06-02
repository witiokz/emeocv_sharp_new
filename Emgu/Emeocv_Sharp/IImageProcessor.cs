using Emgu.CV;
using Emgu.CV.Structure;
using System.Collections.Generic;

namespace Emeocv_Sharp
{
    public interface IImageProcessor
    {
        void SetInput(Image<Bgr, byte> img);
        void Process();
        List<Mat> GetOutput();

        void DebugWindow(bool bval = true);
        void DebugSkew(bool bval = true);
        void DebugEdges(bool bval = true);
        void DebugDigits(bool bval = true);
        void ShowImage();
    }
}