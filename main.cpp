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


int main()
{
    time_t t; 
    srand((unsigned) time(&t));
    printf("Neural Network\n");
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
    //Training
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
    
    //Testing
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