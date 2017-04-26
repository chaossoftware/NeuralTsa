﻿using MathLib.IO;
using MathLib.MathMethods.Lyapunov;
using MathLib.Transform;
using System;
using System.Globalization;
using System.IO;
using System.Text;

namespace NeuralNetwork {
    class NeuralOutput {

        public static string FileName;
        public static string OutDirectory;
        
        public static string BasePath { get { return OutDirectory + "\\" + FileName; } }
        public static string LogFileName { get { return OutDirectory + "\\log.txt"; } }
        public static string SignalPlotFileName { get { return OutDirectory + "\\" + FileName + "_signal.png"; } }
        public static string PoincarePlotFileName { get { return OutDirectory + "\\" + FileName + "_poincare.png"; } }
        public static string ReconstructedSignalPlotFileName { get { return OutDirectory + "\\" + FileName + "_reconstructed_signal.png"; } }
        public static string ReconstructedPoincarePlotFileName { get { return OutDirectory + "\\" + FileName + "_reconstructed_poincare.png"; } }
        public static string PredictedSignalPlotFileName { get { return OutDirectory + "\\" + FileName + "_reconstructed_signal.png"; } }
        public static string NetworkPlotPlotFileName { get { return OutDirectory + "\\" + FileName + "_network_plot.png"; } }
        public static string LeInTimeFileName { get { return BasePath + "_leInTime.le"; } }


        public static bool saveModel = true;
        public static int modelPts = 100000;

        public static bool saveWav = true;
        public static int wavLengthSec = 2;

        public static int predictedSignalPts;
        public static int predictedPoincarePts = 100000;

        public static bool saveLeInTime = true;

        public static void Init(string fileName) {
            OutDirectory = fileName + "_out";
            if (!Directory.Exists(OutDirectory))
                Directory.CreateDirectory(OutDirectory);

            FileName = fileName.Split('\\')[fileName.Split('\\').Length - 1];

            Logger.Init(LogFileName);
        }


        public static void SaveDebugInfoToFile(double ebest, BenettinResult result, double _le, double[] bbest, double[,] abest, int n, int d) {
            
            StringBuilder debug = new StringBuilder();

            debug.AppendFormat(CultureInfo.InvariantCulture, "Training error: {0:0.#####e-0}\n\n", ebest);
            debug.Append(result.GetInfo());
            debug.AppendFormat(CultureInfo.InvariantCulture, "Largest Lyapunov exponent: {0:F5}\n", _le);

            debug.AppendFormat(CultureInfo.InvariantCulture, "\nBias: {0:F8}\n\n", bbest[0]);
            
            for (int i = 1; i <= n; i++)
                debug.AppendFormat("Neuron {0} :\t\t\t", i);
            debug.Append("\n");
        
            for (int j = 0; j <= d; j++) {
                for (int i = 1; i <= n; i++)
                    debug.AppendFormat(CultureInfo.InvariantCulture, "{0:F8}\t\t", abest[i, j]);
                debug.Append("\n");
            }
            debug.Append("\n");

            for (int i = 1; i <= n; i++)
                debug.AppendFormat(CultureInfo.InvariantCulture, "{0:F8}\t\t", bbest[i]);
            debug.Append("\n-----------------------------------------------");

            Logger.LogInfo(debug.ToString(), true);
        }


        /// <summary>
        /// Create file with attractor 3D model in PLY format
        /// </summary>
        /// <param name="xt">array of points X coordinates</param>
        /// <param name="yt">array of points Y coordinates</param>
        /// <param name="zt">array of points Z coordinates</param>
        public static void Create3dModelFile(double[] xt, double[] yt, double[] zt) {
            if (!saveModel) 
                return;

            string filePath = BasePath + "_model.ply";
            Model3D.Create3dModelFile(filePath, xt, yt, zt);
        }


        /// <summary>
        /// Create file "sound" of attractor in WAV format
        /// </summary>
        /// <param name="yt">Y coordinates of attractor points</param>
        public static void CreateWavFile(double[] yt) {

            if (!saveWav) 
                return;

            long pts = yt.Length;
            string filePath = BasePath + "_sound.wav";

            Sound.CreateWavFile(filePath, yt);
        }


        /// <summary>
        /// Create file with predicted points coordinates
        /// Plot predicted signal
        /// </summary>
        /// <param name="predictedPoints">predicted points array</param>
        public static void CreatePredictedDataFile(double[] predictedPoints) {

            StringBuilder prediction = new StringBuilder();

            foreach (double predictedPoint in predictedPoints) 
                prediction.AppendFormat(CultureInfo.InvariantCulture, "{0:F10}\n", predictedPoint);

            DataWriter.CreateDataFile(BasePath + ".predict", prediction.ToString());
        }


        public static void CreateLeInTimeFile(double[,] leInTime) {

            if (!saveLeInTime) return;

            StringBuilder le = new StringBuilder();

            for (int i = 0; i < leInTime.GetLength(1); i++) {
                for (int j = 0; j < leInTime.GetLength(0); j++)
                    le.AppendFormat(CultureInfo.InvariantCulture, "{0:F5}\t", leInTime[j, i]);
                le.Append("\n");
            }

            DataWriter.CreateDataFile(LeInTimeFileName, le.ToString());
        }
    }


    public class Logger {

        private static string LogFile;


        /// <summary>
        /// Logger initialization:
        /// - Recreation of file with log
        /// - Setting name for log-file
        /// </summary>
        /// <param name="fileName"></param>
        public static void Init(string fileName) {
            File.Delete(fileName);
            File.Create(fileName).Close();
            LogFile = fileName;
        }

        public static void LogInfo(string info, bool withTimestamp = false) {
            if (withTimestamp)
                info = GetCurrentTime() + info;
            using (StreamWriter file = new StreamWriter(LogFile, true)) {
                file.WriteLine(info + "\n\n");
            }
        }


        /// <summary>
        /// Get current date-time
        /// </summary>
        /// <returns>current date-time in format: "ShortDate - LongTime"</returns>
        private static string GetCurrentTime() {
            return DateTime.Now.ToShortDateString() + " - " + DateTime.Now.ToLongTimeString() + "\n";
        }
    }
}