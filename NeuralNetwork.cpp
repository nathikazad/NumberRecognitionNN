#include "NeuralNetwork.h"
NeuralNetwork::NeuralNetwork(int number_of_inputs, int number_of_neurons, int number_of_outputs)
{
	this->number_of_inputs=number_of_inputs;
	this->number_of_neurons=number_of_neurons;
	this->number_of_outputs=number_of_outputs;
	this->neuron_layer=new Layer(number_of_inputs,number_of_neurons);
	this->output_layer=new Layer(number_of_neurons,number_of_outputs);
}
float* NeuralNetwork::compute(float* inputs)
{
	return this->output_layer->compute(this->neuron_layer->compute(inputs));
}
void NeuralNetwork::train(float* inputs, float* outputs)
{
	this->output_layer->train_as_output(this->neuron_layer->output_vector, outputs);
	this->neuron_layer->train_as_hidden(inputs, this->output_layer);
}

void NeuralNetwork::debug()
{
	printf("Neuron Layer\n");
	this->neuron_layer->debug();
	printf("Output Layer\n");
	this->output_layer->debug();
}
