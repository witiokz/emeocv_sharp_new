using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace EmeocvSharp
{
    public class Config
    {
        public const string PathToOutput = @"C:\Users\user\Desktop\output";

        public int rotationDegrees { get
            {
                return GetConfigInt("rotationDegrees");
            }
        }
        public float ocrMaxDist
        {
            get
            {
                return GetConfigFloat("ocrMaxDist");
            }
        }
        public int digitMinHeight
        {
            get
            {
                return GetConfigInt("digitMinHeight");
            }
        }
        public int digitMaxHeight
        {
            get
            {
                return GetConfigInt("digitMaxHeight");
            }
        }
        public int digitYAlignment
        {
            get
            {
                return GetConfigInt("digitYAlignment");
            }
        }
        public int cannyThreshold1
        {
            get
            {
                return GetConfigInt("cannyThreshold1");
            }
        }
        public int cannyThreshold2
        {
            get
            {
                return GetConfigInt("cannyThreshold2");
            }
        }
        public string trainingDataFilename
        {
            get
            {
                return GetConfig("trainingDataFilename");
            }
        }

        private string GetConfig(string key)
        {
            return System.Configuration.ConfigurationManager.AppSettings[key];
        }
        private int GetConfigInt(string key)
        {
            var value =  System.Configuration.ConfigurationManager.AppSettings[key];
            if (!string.IsNullOrEmpty(value))
            {
                return int.Parse(value);
            }
            return 0;
        }
        private float GetConfigFloat(string key)
        {
            var value = System.Configuration.ConfigurationManager.AppSettings[key];
            if (!string.IsNullOrEmpty(value))
            {
                return float.Parse(value);
            }
            return 0;
        }
    }
}
