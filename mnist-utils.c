/**
 * @file mnist-utils.h
 * @brief Utitlies for handling the MNIST files
 * @see http://yann.lecun.com/exdb/mnist/
 * @author Matt Lind
 * @date July 2015
 */

#include <stdint.h>
#include <stdio.h>


#define MNIST_TRAINING_SET_IMAGE_FILE_NAME "data/train-images-idx3-ubyte" ///< MNIST image training file in the data folder
#define MNIST_TRAINING_SET_LABEL_FILE_NAME "data/train-labels-idx1-ubyte" ///< MNIST label training file in the data folder

#define MNIST_TESTING_SET_IMAGE_FILE_NAME "data/t10k-images-idx3-ubyte"  ///< MNIST image testing file in the data folder
#define MNIST_TESTING_SET_LABEL_FILE_NAME "data/t10k-labels-idx1-ubyte"  ///< MNIST label testing file in the data folder



#define MNIST_MAX_TRAINING_IMAGES 60000                     ///< number of images+labels in the TRAIN file/s
#define MNIST_MAX_TESTING_IMAGES 10000                      ///< number of images+labels in the TEST file/s
#define MNIST_IMG_WIDTH 28                                  ///< image width in pixel
#define MNIST_IMG_HEIGHT 28                                 ///< image height in pixel



typedef struct MNIST_ImageFileHeader MNIST_ImageFileHeader;
typedef struct MNIST_LabelFileHeader MNIST_LabelFileHeader;

typedef struct MNIST_Image MNIST_Image;
typedef uint8_t MNIST_Label;



/**
 * @brief Data block defining a MNIST image
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */
struct MNIST_Image{
    uint8_t pixel[28*28];
};




/**
 * @brief Data block defining a MNIST image file header
 * @attention The fields in this structure are not used.
 * What matters is their byte size to move the file pointer
 * to the first image.
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_ImageFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
};




/**
 * @brief Data block defining a MNIST label file header
 * @attention The fields in this structure are not used.
 * What matters is their byte size to move the file pointer
 * to the first label.
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_LabelFileHeader{
    uint32_t magicNumber;
    uint32_t maxImages;
};




/**
 * @brief Read MNIST IMAGE file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

FILE *openMNISTImageFile(char *fileName);




/**
 * @brief Read MNIST label file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

FILE *openMNISTLabelFile(char *fileName);



/**
 * @brief Returns the next image in given MNIST image file
 */

MNIST_Image getImage(FILE *imageFile);




/**
 * @brief Returns the next label in given MNIST label file
 */

MNIST_Label getLabel(FILE *labelFile);

void getDesiredOutputVector(FILE *labelFile, float* correct_output);

void getNormalizedPixels(FILE *imageFile, float *normalized_pixels);




/**
 * @details Reverse byte order in 32bit numbers
 * MNIST files contain all numbers in reversed byte order,
 * and hence must be reversed before using
 */

uint32_t flipBytes(uint32_t n){
    
    uint32_t b0,b1,b2,b3;
    
    b0 = (n & 0x000000ff) <<  24u;
    b1 = (n & 0x0000ff00) <<   8u;
    b2 = (n & 0x00ff0000) >>   8u;
    b3 = (n & 0xff000000) >>  24u;
    
    return (b0 | b1 | b2 | b3);
    
}




/**
 * @details Read MNIST image file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

void readImageFileHeader(FILE *imageFile, MNIST_ImageFileHeader *ifh){
    
    ifh->magicNumber =0;
    ifh->maxImages   =0;
    ifh->imgWidth    =0;
    ifh->imgHeight   =0;
    
    fread(&ifh->magicNumber, 4, 1, imageFile);
    ifh->magicNumber = flipBytes(ifh->magicNumber);
    
    fread(&ifh->maxImages, 4, 1, imageFile);
    ifh->maxImages = flipBytes(ifh->maxImages);
    
    fread(&ifh->imgWidth, 4, 1, imageFile);
    ifh->imgWidth = flipBytes(ifh->imgWidth);
    
    fread(&ifh->imgHeight, 4, 1, imageFile);
    ifh->imgHeight = flipBytes(ifh->imgHeight);
}




/**
 * @details Read MNIST label file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

void readLabelFileHeader(FILE *imageFile, MNIST_LabelFileHeader *lfh){
    
    lfh->magicNumber =0;
    lfh->maxImages   =0;
    
    fread(&lfh->magicNumber, 4, 1, imageFile);
    lfh->magicNumber = flipBytes(lfh->magicNumber);
    
    fread(&lfh->maxImages, 4, 1, imageFile);
    lfh->maxImages = flipBytes(lfh->maxImages);
    
}




/**
 * @details Open MNIST image file and read header info
 * by reading the header info, the read pointer
 * is moved to the position of the 1st IMAGE
 */

FILE *openMNISTImageFile(char *fileName){

    FILE *imageFile;
    imageFile = fopen (fileName, "rb");
    if (imageFile == NULL) {
        printf("Abort! Could not fine MNIST IMAGE file: %s\n",fileName);
        exit(0);
    }

    MNIST_ImageFileHeader imageFileHeader;
    readImageFileHeader(imageFile, &imageFileHeader);
    
    return imageFile;
}




/**
 * @details Open MNIST label file and read header info
 * by reading the header info, the read pointer
 * is moved to the position of the 1st LABEL
 */

FILE *openMNISTLabelFile(char *fileName){
    
    FILE *labelFile;
    labelFile = fopen (fileName, "rb");
    if (labelFile == NULL) {
        printf("Abort! Could not find MNIST LABEL file: %s\n",fileName);
        exit(0);
    }

    MNIST_LabelFileHeader labelFileHeader;
    readLabelFileHeader(labelFile, &labelFileHeader);
    
    return labelFile;
}




/**
 * @details Returns the next image in the given MNIST image file
 */

MNIST_Image getImage(FILE *imageFile){
    
    MNIST_Image img;
    size_t result;
    result = fread(&img, sizeof(img), 1, imageFile);
    if (result!=1) {
        printf("\nError when reading IMAGE file! Abort!\n");
        exit(1);
    }
    
    return img;
}


void getNormalizedPixels(FILE *imageFile, float *normalized_pixels)
{
    MNIST_Image img = getImage(imageFile);
    uint8_t *pixels=img.pixel;
    int i;
    for(i=0;i<784;i++)
        normalized_pixels[i]=(float)pixels[i]/255.0;
}

/**
 * @details Returns the next label in the given MNIST label file
 */

MNIST_Label getLabel(FILE *labelFile){
    
    MNIST_Label lbl;
    size_t result;
    result = fread(&lbl, sizeof(lbl), 1, labelFile);
    if (result!=1) {
        printf("\nError when reading LABEL file! Abort!\n");
        // exit(1);
    }
    
    return lbl;
}

void getDesiredOutputVector(FILE *labelFile, float* correct_output)
{
    int i;
    for(i=0;i<10;i++)
    {
        correct_output[i]=0;
    }
    correct_output[getLabel(labelFile)]=1.0;
}

