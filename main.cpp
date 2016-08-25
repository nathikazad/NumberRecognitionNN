#include <stdio.h>
#include <time.h> 
#include "NeuralNetwork.h"
#include "mnist-utils.h"
#define iter 1000000
#define epoch 10
int max_index(float *array)
{
	int max_index=0;
	for(int i=1;i<10;i++)
		if(array[i]>array[max_index])
			max_index=i;
	return max_index;
}


// int main()
// {
// 	time_t t;
// 	srand((unsigned) time(&t));
// 	NeuralNetwork NN(1, 3, 1);
// 	float *output;
// 	float input[1];
// 	float desired_output[1];
// 	float discrete_output;
// 	float correct = 0;
// 	int i;
// 	for(i=0;i<iter;i++)
// 	{
// 		input[0]=(float)(rand()%1000)/1000;
// 		// input[1]=(float)(rand()%1000)/1000;
// 		desired_output[0]=0.0;
// 		if(input[0]>0.5)
// 			desired_output[0]=1.0;
// 		output=NN.compute(input);
// 		if(output[0]>0.5)
// 			discrete_output=1.0;
// 		else
// 			discrete_output=0.0;
		
// 		if(discrete_output==desired_output[0])
// 			correct++;

// 		NN.train(input,desired_output);
// 		printf("x=%.3f o=%.1f do=%.1f %s \n", input[0], discrete_output, desired_output[0], (discrete_output==desired_output[0]) ? "CORRECT" : "WRONG");
// 		// NN.debug();
		
// 	}
// 	printf("Correct:%f\n",correct/iter);
// }

int main()
{
	time_t t; 
	srand((unsigned) time(&t));
	printf("Neural Net\n");
	NeuralNetwork NN(784,1000,10);
	float *output;
	float pixels[784];
	float correct_output[10];
	FILE *imageFile, *labelFile;
	char training_image_file[]="data/train-images-idx3-ubyte";
	char training_label_file[]="data/train-labels-idx1-ubyte";

	char testing_image_file[]="data/t10k-images-idx3-ubyte";
	char testing_label_file[]="data/t10k-labels-idx1-ubyte";
    float correct_predictions = 0;
    for(int j=0;j<epoch;j++)
    {
		printf("Epoch %d\n", j);
		imageFile = openMNISTImageFile(training_image_file);
    	labelFile = openMNISTLabelFile(training_label_file);
		for(int i=0;i<MNIST_MAX_TRAINING_IMAGES;i++)
		{
	        if(i%10000==0)
	        	printf("%d images\n",i);
	        getNormalizedPixels(imageFile, pixels);
	        getDesiredOutputVector(labelFile, correct_output);
			output=NN.compute(pixels);
			NN.train(pixels,correct_output);
		}
	}
	fclose(imageFile);
    fclose(labelFile);
	
	imageFile = openMNISTImageFile(testing_image_file);
	labelFile = openMNISTLabelFile(testing_label_file);
	for(int i=0;i<MNIST_MAX_TESTING_IMAGES;i++)
	{
        getNormalizedPixels(imageFile, pixels);
        getDesiredOutputVector(labelFile, correct_output);
		output=NN.compute(pixels);
		if(max_index(output)==max_index(correct_output))
			correct_predictions++;
	}
	fclose(imageFile);
    fclose(labelFile);
	printf("Correct Predictions:%f\n",correct_predictions/(MNIST_MAX_TESTING_IMAGES));
	return 0;
}