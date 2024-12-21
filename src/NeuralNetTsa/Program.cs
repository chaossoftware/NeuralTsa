using NeuralNetTsa.Configuration;
using System;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Threading;

namespace NeuralNetTsa;

internal class Program
{
    static void Main(string[] args)
    {
        Console.Title = "Neural Net Time Series Analyzer";
        Console.OutputEncoding = System.Text.Encoding.Unicode;
        Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");
        Thread.CurrentThread.CurrentUICulture = new CultureInfo("en-US");

        Config config = new();
        FileVersionInfo versionInfo = FileVersionInfo.GetVersionInfo(Assembly.GetExecutingAssembly().Location);

        foreach (var dataFile in config.Files)
        {
            Console.Clear();
            Console.WriteLine($"Version: {versionInfo.ProductVersion}");
            Console.WriteLine($"File: {dataFile.FileName}");
            FileProcessor.ProcessFile(config, dataFile);
        }
    }
}
