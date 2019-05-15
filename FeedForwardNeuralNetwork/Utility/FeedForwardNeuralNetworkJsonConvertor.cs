﻿using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using FeedForwardNeuralNetwork.Core;
using Newtonsoft.Json.Linq;
using FeedForwardNeuralNetwork = FeedForwardNeuralNetwork.Core.FeedForwardNeuralNetwork;

namespace FeedForwardNeuralNetwork.Utility
{
    internal class FeedForwardNeuralNetworkJsonConvertor : JsonConverter
    {
        private Core.FeedForwardNeuralNetwork _network;
        private string _data;

        public FeedForwardNeuralNetworkJsonConvertor(string data)
        {
            _data = data;
        }

        public FeedForwardNeuralNetworkJsonConvertor(Core.FeedForwardNeuralNetwork network)
        {
            _network = network;
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            int inputLayerNeuronsCount = _network.InputLayer.Neurons.Length;
            JArray inputNeurons = new JArray();
            for (int i = 0; i < inputLayerNeuronsCount; i++)
            {
                inputNeurons.Add(_network.InputLayer.Neurons[i].Bias);
                inputNeurons.Add(_network.InputLayer.Neurons[i].Delta);
                inputNeurons.Add(_network.InputLayer.Neurons[i].Value);
                JArray inputDendrite = new JArray();
                for (int j = 0; j < _network.InputLayer.Neurons[i].Weights.Length; j++)
                {
                    inputDendrite.Add(_network.InputLayer.Neurons[i].Weights[j].Value);
                }
                inputNeurons.Add(inputDendrite);
            }
            JArray inputLayerObject = new JArray { { inputNeurons } };

            int hiddenLayerNeuronsCount = _network.HiddenLayer.Neurons.Length;
            JArray hiddenNeurons = new JArray();
            for (int i = 0; i < hiddenLayerNeuronsCount; i++)
            {
                hiddenNeurons.Add(_network.HiddenLayer.Neurons[i].Bias);
                hiddenNeurons.Add(_network.HiddenLayer.Neurons[i].Delta);
                hiddenNeurons.Add(_network.HiddenLayer.Neurons[i].Value);
                JArray hiddenDendrite = new JArray();
                for (int j = 0; j < _network.HiddenLayer.Neurons[i].Weights.Length; j++)
                {
                    hiddenDendrite.Add(_network.HiddenLayer.Neurons[i].Weights[j].Value);
                }
                hiddenNeurons.Add(hiddenDendrite);
            }
            JArray hiddenLayerObject = new JArray { { hiddenNeurons } };


            int outputLayerNeuronsCount = _network.OutputLayer.Neurons.Length;
            JArray outputNeurons = new JArray();
            for (int i = 0; i < outputLayerNeuronsCount; i++)
            {
                outputNeurons.Add(_network.OutputLayer.Neurons[i].Bias);
                outputNeurons.Add(_network.OutputLayer.Neurons[i].Delta);
                outputNeurons.Add(_network.OutputLayer.Neurons[i].Value);
                JArray outputDendrite = new JArray();
                for (int j = 0; j < _network.OutputLayer.Neurons[i].Weights.Length; j++)
                {
                    outputDendrite.Add(_network.OutputLayer.Neurons[i].Weights[j].Value);
                }
                outputNeurons.Add(outputDendrite);
            }
            JArray outputLayerObject = new JArray() { { outputNeurons } };

            JObject networkObject = new JObject { { "Network", new JArray { inputLayerObject, hiddenLayerObject, outputLayerObject } }, { "LearningRate", _network.LearningRate } };


            networkObject.WriteTo(writer);
        }

        public Core.FeedForwardNeuralNetwork DeserializeData()
        {
            JObject json = JObject.Parse(_data);

            JToken networkObject = json["Network"];
            JToken inputLayerObject = networkObject[0];
            JToken hiddenLayerObject = networkObject[1];
            JToken outputLayerObject = networkObject[2];

            /***********************input layer*********************************/
            int count = inputLayerObject[0].Count() / 4;
            double[] biases = new double[count];
            double[] deltas = new double[count];
            double[] values = new double[count];
            List<double[]> dendritesList = new List<double[]>();

            for (int i = 0; i < count; i++)
            {
                biases[i] = (double)inputLayerObject[0][i * 4];
                deltas[i] = (double)inputLayerObject[0][i * 4 + 1];
                values[i] = (double)inputLayerObject[0][i * 4 + 2];
                dendritesList.Add(new double[0]);
            }

            Layer inputLayer = new Layer(biases, deltas, values, dendritesList);
            /***********************input layer*********************************/

            /***********************hidden layer*********************************/
            count = hiddenLayerObject[0].Count() / 4;
            biases = new double[count];
            deltas = new double[count];
            values = new double[count];
            dendritesList = new List<double[]>();

            for (int i = 0; i < count; i++)
            {
                biases[i] = (double)hiddenLayerObject[0][i * 4];
                deltas[i] = (double)hiddenLayerObject[0][i * 4 + 1];
                values[i] = (double)hiddenLayerObject[0][i * 4 + 2];
                JToken dendrites = hiddenLayerObject[0][i * 4 + 3];
                double[] dendritesArray = new double[dendrites.Count()];
                for (int j = 0; j < dendrites.Count(); j++)
                {
                    dendritesArray[j] = (double)dendrites[j];
                }
                dendritesList.Add(dendritesArray);
            }

            Layer hiddenLayer = new Layer(biases, deltas, values, dendritesList);
            /***********************hidden layer*********************************/

            /***********************output layer*********************************/
            count = outputLayerObject[0].Count() / 4;
            biases = new double[count];
            deltas = new double[count];
            values = new double[count];
            dendritesList = new List<double[]>();

            for (int i = 0; i < count; i++)
            {
                biases[i] = (double)outputLayerObject[0][i * 4];
                deltas[i] = (double)outputLayerObject[0][i * 4 + 1];
                values[i] = (double)outputLayerObject[0][i * 4 + 2];
                JToken dendrites = outputLayerObject[0][i * 4 + 3];
                double[] dendritesArray = new double[dendrites.Count()];
                for (int j = 0; j < dendrites.Count(); j++)
                {
                    dendritesArray[j] = (double)dendrites[j];
                }
                dendritesList.Add(dendritesArray);
            }

            Layer outputLayer = new Layer(biases, deltas, values, dendritesList);
            /***********************output layer*********************************/

            /***********************Learning Rate*******************************/
            double learningRate = (double)json["LearningRate"];
            /***********************Learning Rate*******************************/

            Core.FeedForwardNeuralNetwork net = new Core.FeedForwardNeuralNetwork(inputLayer, hiddenLayer, outputLayer, learningRate);
            return net;
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            throw new NotImplementedException();
        }

        public override bool CanConvert(Type objectType)
        {
            return true;
        }

        public override bool CanRead => false;
    }
}