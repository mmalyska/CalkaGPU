using Alea;
using Alea.CSharp;
using System;
using System.Diagnostics;
using System.Linq;
using Alea.Parallel;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using System.Globalization;

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
            ////var result = CalkaGPU(n);
            //stopwatch.Stop();

            //Console.WriteLine(result);
            //Console.WriteLine("Time: " + stopwatch.Elapsed.TotalSeconds);

            //result = 0;

            //stopwatch.Reset();

            stopwatch.Start();
            var result = CalkaCPU(n, 0, 5000);
            result += CalkaCPU(n, 5000, n);
            stopwatch.Stop();

            Console.WriteLine(result);
            Console.WriteLine("Time: " + stopwatch.Elapsed.TotalSeconds);

            stopwatch.Reset();

            stopwatch.Start();
            var result1 = CalkaRozproszone(n).Result;
            stopwatch.Stop();

            Console.WriteLine(result1);
            Console.WriteLine("Time: " + stopwatch.Elapsed.TotalSeconds);
            Console.ReadLine();
        }

        private static double CalkaCPU(int n, int start, int stop)
        {
            double result = 0;
            double ub = 1;
            double lb = 0;
            var dxy = (ub - lb) / n;
            for (int yi = start; yi < stop; yi++)
            {
                double y = (yi + 0.5) * dxy;
                for (int xi = 0; xi < n; xi++)
                {
                    double x = (xi + 0.5) * dxy;
                    result += Calka(x, y) * dxy * dxy;
                }
            }
            return result;
        }

        [GpuManaged]
        public static double CalkaGPU(int n)
        {
            var gpu = Gpu.Default;
            var lp = LaunchParam1d(n);
            double[] result = new double[lp.GridDim.x];
            double ub = 1;
            double lb = 0;
            double d = (ub - lb) / n;
            gpu.Launch(Kernel, lp, result, n, lb, ub, d);
            return result.Sum();
        }

        private async static Task<double> CalkaRozproszone(int n)
        {
            double result = 0;
            double ub = 0;
            double lb = 0;
            double chunk = (double)1 / (double)ClientsURLs.Length;

            ub = chunk;        
            foreach(var clientUrl in ClientsURLs)
            {
                HttpClient client = new HttpClient();
                client.BaseAddress = new Uri(clientUrl);
                client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

                HttpResponseMessage response = await client.GetAsync(string.Format("api/calka/wynik/cpu?lb={0}&ub={1}&n={2}", lb.ToString().Replace(",", "."), ub.ToString().Replace(",", "."), n));

                if (response.IsSuccessStatusCode)
                {
                    result += double.Parse(response.Content.ReadAsStringAsync().Result, CultureInfo.InvariantCulture);
                    lb += chunk;
                    ub += chunk;
                }
                else
                    throw new Exception("Wystąpił błąd podczas obliczania całki!");
            }

            return result;
        }

        public static void Kernel(double[] result, int n, double lb, double ub, double d)
        {
            var temp = __shared__.Array<double>(BlockSize);
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var step = gridDim.x * blockDim.x;
            for (int yi = start; yi < n; yi += step)
            {
                double y = (yi + 0.5) * d;
                double intSum = 0;
                for (int xi = 0; xi < n; xi++)
                {
                    double x = (xi + 0.5) * d;
                    intSum += CalkaGPU(x, y);
                }
                temp[threadIdx.x] += intSum * d * d;
            }

            DeviceFunction.SyncThreads();

            if (threadIdx.x == 0)
            {
                for (int i = 0; i < BlockSize; i++)
                {
                    result[blockIdx.x] += temp[i];
                }
            }
        }

        private static double Calka(double x, double y)
        {   //cos(x) + sin(x*y) + 0.1*x^2 - 2*y^2
            return Math.Cos(x) + Math.Sin(x * y) + 0.1 * x * x - 2 * y * y;
        }

        private static double CalkaGPU(double x, double y)
        {   //cos(x) + sin(x*y) + 0.1*x^2 - 2*y^2
            return DeviceFunction.Cos(x) + DeviceFunction.Sin(x * y) + 0.1 * x * x - 2 * y * y;
        }

        private static LaunchParam LaunchParam1d(int n)
        {
            var blockSize = new dim3(BlockSize);
            var gridSize = new dim3(DivUp(n, BlockSize));
            return new LaunchParam(gridSize, blockSize);
        }

        private static int DivUp(int num, int den)
        {
            return (num + den - 1) / den;
        }
    }
}
