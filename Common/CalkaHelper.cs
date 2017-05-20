using Alea;
using Alea.CSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Common
{
    public class CalkaHelper
    {
        private const int BlockSize = 32;

        public double CalkaCPU(double lb, double ub, int n)
        {
            double result = 0;

            var dxy = (ub - lb) / n;
            for (int yi = 0; yi < n; yi++)
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
        public double CalkaGPU(double lb, double ub, int n)
        {
            var gpu = Gpu.Default;
            var lp = LaunchParam1d(n);
            double[] result = new double[lp.GridDim.x];

            double d = (ub - lb) / n;
            gpu.Launch(Kernel, lp, result, n, lb, ub, d);
            return result.Sum();
        }

        public void Kernel(double[] result, int n, double lb, double ub, double d)
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

        private double Calka(double x, double y)
        {   //cos(x) + sin(x*y) + 0.1*x^2 - 2*y^2
            return Math.Cos(x) + Math.Sin(x * y) + 0.1 * x * x - 2 * y * y;
        }

        private double CalkaGPU(double x, double y)
        {   //cos(x) + sin(x*y) + 0.1*x^2 - 2*y^2
            return DeviceFunction.Cos(x) + DeviceFunction.Sin(x * y) + 0.1 * x * x - 2 * y * y;
        }

        private LaunchParam LaunchParam1d(int n)
        {
            var blockSize = new dim3(BlockSize);
            var gridSize = new dim3(DivUp(n, BlockSize));
            return new LaunchParam(gridSize, blockSize);
        }

        private int DivUp(int num, int den)
        {
            return (num + den - 1) / den;
        }
    }
}
