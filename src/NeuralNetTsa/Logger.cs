using System;
using System.IO;

namespace NeuralNetTsa;

public static class Logger
{
    private static string LogFile;

    /// <summary>
    /// Logger initialization:
    /// - Recreation of file with log
    /// - Setting name for log-file
    /// </summary>
    /// <param name="fileName"></param>
    public static void Init(string fileName)
    {
        File.Delete(fileName);
        File.Create(fileName).Close();
        LogFile = fileName;
    }

    public static void LogInfo(string info, bool withTimestamp = false)
    {
        if (withTimestamp)
        {
            info = $"{DateTime.Now}\n{info}";
        }

        using (StreamWriter file = new StreamWriter(LogFile, true))
        {
            file.WriteLine(info + "\n\n");
        }
    }
}
