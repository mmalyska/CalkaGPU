using System;
using System.Diagnostics;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using System.Globalization;
using Common;

namespace CalkaGPU
{
    class Program
    {
        private const int BlockSize = 32;
        private static string[] ClientsURLs = new string[] { "http://localhost:8081", "http://localhost:8082" };

        static void Main(string[] args)
        {
            int n = 10000;
            Stopwatch stopwatch = new Stopwatch();

            //stopwatch.Start();
            //var result1 = CalkaHelper.CalkaGPU(n);
            //stopwatch.Stop();

            //Console.WriteLine(result1);
            //Console.WriteLine("Time: " + stopwatch.Elapsed.TotalSeconds);
       
            stopwatch.Reset();

            stopwatch.Start();
            var result2 = CalkaHelper.CalkaCPU(0, n, n);
            stopwatch.Stop();

            Console.WriteLine(result2);
            Console.WriteLine("Time: " + stopwatch.Elapsed.TotalSeconds);

            stopwatch.Reset();

            stopwatch.Start();
            var result3 = CalkaHelper.CalkaRozproszone(ClientsURLs, n);
            stopwatch.Stop();

            Console.WriteLine(result3);
            Console.WriteLine("Time: " + stopwatch.Elapsed.TotalSeconds);
            Console.ReadLine();
        }
    }
}
