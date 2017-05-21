using Alea;
using Alea.CSharp;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    public static class CalkaHelper
    {
        private const int BlockSize = 32;

        public static double CalkaCPU(int start, int stop, int n)
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

        public static double CalkaRozproszone(string[] clients, int n)
        {
            var tasks = new List<Task<HttpResponseMessage>>();
            double result = 0;

            int start, stop, chunk;

            start = 0;
            chunk = n / clients.Length;
            stop = chunk;

            foreach (var url in clients)
            {
                HttpClient client = new HttpClient() { BaseAddress = new Uri(url) };
                tasks.Add(client.GetAsync(string.Format("api/calka/wynik/cpu?start={0}&stop={1}&n={2}", start, stop, n)));

                start += chunk;
                stop += chunk;
            }

            Task.WaitAll(tasks.ToArray());

            foreach (var res in tasks)
            {
                if (res.Result.IsSuccessStatusCode)
                    result += double.Parse(res.Result.Content.ReadAsStringAsync().Result, CultureInfo.InvariantCulture);
                else
                    throw new Exception("Wystąpił błąd podczas obliczania całki!");
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
