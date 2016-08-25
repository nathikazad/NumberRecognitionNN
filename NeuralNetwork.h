#include <stdio.h>
#include "Layer.h"
class NeuralNetwork
{
	public:
		NeuralNetwork(int number_of_inputs, int number_of_neurons, int number_of_outputs);
		int number_of_inputs;
		int number_of_neurons;
		int number_of_outputs;
		Layer *neuron_layer;
		Layer *output_layer;
		float* compute(float* inputs);
		void train(float* inputs, float* outputs);
		void debug();
};