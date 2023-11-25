using System;
using System.Diagnostics;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        readonly double[][,] _weights;
        readonly double[][] _layers;
        readonly double[][] _biases;
        private readonly Stopwatch _watch = new Stopwatch();

        public StudentNetwork(int[] structure)
        {
            _weights = new double[structure.Length - 1][,]; // Матрица весов между слоями
            _biases = new double[structure.Length - 1][]; // Массив сдвигов
            _layers = new double[structure.Length][]; // layers[i] -> layers[i+1] //400 700 20 4

            for (int i = 0; i < structure.Length; i++)
            {
                _layers[i] = new double[structure[i]];
                if (i != structure.Length - 1)
                {
                    _weights[i] = new double[structure[i], structure[i + 1]];
                    _biases[i] = new double[structure[i + 1]];
                    FillRandomMatrix(_weights[i]);
                    FillRandomArray(_biases[i]);
                }
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            return _Train(sample, acceptableError).Item1;
        }

        private (int, double) _Train(Sample sample, double acceptableError)
        {
            const double NU = 1.0;
            int iterations = 0;

            double[] result = Compute(sample.input);
            double[] expected = sample.Output;
            double error = GetError(result, expected);

            while (error > acceptableError)
            {
                double[][] delta = new double[_layers.Length][];

                for (int i = _layers.Length - 1; i >= 0; i--)
                {
                    delta[i] = new double[_layers[i].Length];

                    for (int j = 0; j < _layers[i].Length; j++)
                    {
                        if (i == _layers.Length - 1)
                        {
                            delta[i][j] = _layers[i][j] * (1 - _layers[i][j]) * (_layers[i][j] - expected[j]);
                        }
                        else
                        {
                            double sum = 0;

                            for (int k = 0; k < _layers[i + 1].Length; k++)
                            {
                                sum += delta[i + 1][k] * _weights[i][j, k];
                            }

                            delta[i][j] = _layers[i][j] * (1 - _layers[i][j]) * sum;
                        }
                    }
                }

                for (int k = _weights.Length - 1; k >= 0; k--)
                {
                    for (int i = 0; i < _weights[k].GetLength(0); i++)
                    {
                        for (int j = 0; j < _weights[k].GetLength(1); j++)
                        {
                            _weights[k][i, j] -= NU * delta[k + 1][j] * _layers[k][i];
                        }
                    }
                }

                for (int k = _biases.Length - 1; k >= 0; k--)
                {
                    for (int i = 0; i < _biases[k].Length; i++)
                    {
                        _biases[k][i] -= NU * delta[k + 1][i];
                    }
                }

                iterations++;
                result = Compute(sample.input);
                error = GetError(result, expected);
            }

            return (iterations, error);
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            int runnedEpochs = 0;

            _watch.Restart();

            double error = double.PositiveInfinity;

            for (int k = 0; k < epochsCount; k++)
            {
                double errorSum = 0;
                samplesSet.samples.ForEach(sample =>
                {
                    errorSum += _Train(sample, acceptableError).Item2;
                });

                error = errorSum;

                runnedEpochs++;
                OnTrainProgress((runnedEpochs * 1.0) / epochsCount, error, _watch.Elapsed);
            }

            OnTrainProgress(1.0, 5, _watch.Elapsed);
            _watch.Stop();

            return error;
        }

        protected override double[] Compute(double[] input)
        {
            FillLayer(_layers[0], Normalize(input));

            for (int k = 0; k < _layers.Length - 1; k++)
            { 
                for (int j = 0; j < _layers[k+1].Length; j++)
                {
                    double sum = 0;
                    for (int i = 0; i < _layers[k].Length; i++)
                    {
                        sum += _layers[k][i] * _weights[k][i, j];
                    }
                    _layers[k+1][j] = Activate(sum + _biases[k][j]); // Применение функции активации. (передаточная)
                }
            }

            return _layers[_layers.Length - 1];
        }

        private void FillRandomMatrix(double[,] weights)
        {
            Random random = new Random();
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    weights[i, j] = (random.NextDouble() - 0.5); // [-0.5, 0.5]
                }
            }
        }


        private void FillRandomArray(double[] array)
        {
            Random random = new Random();
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = (random.NextDouble() - 0.5); // [-0.5, 0.5]
            }
        }

        private double Activate(double weight, double bias = 0)
        {
            return 1.0 / (1.0 + Math.Exp(-1 * (weight - bias)));
        }

        private double[] Normalize(double[] source, double l = 0, double u = 1) // [0, 200] -> [0, 1]
        {
            double min = 0;
            double max = 200;
            double[] res = (double[])source.Clone();

            for (int i = 0; i < source.Length; i++)
            {
                res[i] = ((source[i] - min) / (max - min)) * (u - l) + l;
            }
            return res;
        }

        private double GetError(double[] res, double[] expect) // Среднее квадратичное отклонение
        {
            double error = 0;
            for (int i = 0; i < res.Length; i++)
            {
                error += Math.Pow(expect[i] - res[i], 2);
            }
            return error / 2;
        }

        private void FillLayer(double[] target, double[] source)
        {
            for (int i = 0; i < target.Length; i++)
            {
                target[i] = source[i];
            }
        }
    }
}