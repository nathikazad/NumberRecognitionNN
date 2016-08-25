#include "Neuron.h"
#define TILEWIDTH 10
Neuron::Neuron(int number_of_inputs)
{
	weights=new float[number_of_inputs+1];
	old_weights=new float[number_of_inputs+1];
	this->number_of_inputs=number_of_inputs;
	for(int i=0; i<number_of_inputs+1; i++)
	{
		weights[i]=(rand()%100)/(float)100/(float)number_of_inputs;
		// printf("%f \n",weights[i]);
	}
}
float Neuron::compute(float* inputs)
{
	float sum=0.0;
	int i,j;
	//#pragma omp parallel for reduction(+:sum)
	for(i=0;i<number_of_inputs/TILEWIDTH;i++)
		for(j=0;j<TILEWIDTH;j++)
		sum+=inputs[i*TILEWIDTH+j]*weights[i*TILEWIDTH+j];
	sum+=1*weights[number_of_inputs];
	return 1/(1+exp(-sum));
}
float Neuron::train_as_output(float* inputs, float actual_output, float desired_output)
{
	
	float dirac=actual_output*(1-actual_output)*(desired_output-actual_output);
	// #pragma omp parallel for
	for(int i=0;i<number_of_inputs;i++)
	{	
		old_weights[i]=weights[i];
		weights[i]=weights[i]+ALPHA*inputs[i]*dirac;
	}
	old_weights[number_of_inputs]=weights[number_of_inputs];
	weights[number_of_inputs]=weights[number_of_inputs]+ALPHA*dirac;
	return dirac;
}
float Neuron::train_as_hidden(float* inputs, float actual_output, float dirac_weight_sum)
{
	float dirac=actual_output*(1-actual_output)*dirac_weight_sum;
	// #pragma omp parallel for
	for(int i=0;i<number_of_inputs;i++)
	{	
		weights[i]=weights[i]+ALPHA*inputs[i]*dirac;
	}
	weights[number_of_inputs]=weights[number_of_inputs]+ALPHA*dirac;
	return dirac;
}
void Neuron::debug()
{
	// printf("\t\t\t Number of inputs:%d\n",number_of_inputs);
	printf("\t\t\t Weights:");
	for(int i=0;i<number_of_inputs;i++)
	{
		printf("%f ",weights[i]);
	}
	printf("Bias:%f",weights[number_of_inputs]);
	printf("\n");
}
