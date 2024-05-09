#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <time.h>

using namespace cv;
using namespace std;
#define KERNEL_HEIGHT 3
#define KERNEL_WIDTH 3

Mat highPassFilter(const Mat& originalImage, const Mat& kernel) {
    // Define padding size
    int paddingSize = (KERNEL_HEIGHT - 1) / 2; // Assuming KERNEL_HEIGHT is odd

    // Create a padded version of the original image
    Mat paddedImage;
    copyMakeBorder(originalImage, paddedImage, paddingSize, paddingSize, paddingSize, paddingSize,
        BORDER_CONSTANT, Scalar(0));
    // Convert the padded image to float
    Mat floatImage;
    paddedImage.convertTo(floatImage, CV_32F); // Convert to float

    Mat highPassImage(originalImage.size(), originalImage.type()); // Create output image

    // Iterate over each pixel in the image
    for (int i = 0; i < originalImage.rows; ++i) {
        for (int j = 0; j < originalImage.cols; ++j) {
            // Compute the sum of element-wise products between the kernel and the corresponding section of the image
            Vec3f sum(0, 0, 0);
            for (int m = 0; m < KERNEL_HEIGHT; ++m) {
                for (int n = 0; n < KERNEL_WIDTH; ++n) {
                    Vec3f pixel = floatImage.at<Vec3f>(i + m, j + n);
                    sum[0] += pixel[0] * kernel.at<float>(m, n);
                    sum[1] += pixel[1] * kernel.at<float>(m, n);
                    sum[2] += pixel[2] * kernel.at<float>(m, n);
                }
            }
            // Store the result in the output image
            highPassImage.at<Vec3b>(i, j) = sum;
        }
    }
    return highPassImage;
}


void parallelHighPassFilter(const Mat& imageData, const Mat& kernel, int rank, int size) {
    if (size == 1) {
        // Only one process, apply high-pass filter directly
        Mat processedImage = highPassFilter(imageData, kernel);
        imshow("Processed Image", processedImage);
        waitKey(0); // Wait for a key press to close the window
        cout << "Result Image Displayed (Single Process)" << endl;
    }
    else if (rank == 0) {
        // Create processed image
        Mat processedImage(imageData.rows, imageData.cols, imageData.type());

        // Get processed parts from processes to combine them
        for (int i = 1; i < size; i++) {
            // Get processed subimage and its position from rank i
            Mat receivedSubImage;
            int y, width, height;
            MPI_Recv(&y, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&width, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&height, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Allocate memory for the received subimage
            receivedSubImage.create(height, width, imageData.type());

            // Get the subimage data
            MPI_Recv(receivedSubImage.data, width * height * imageData.elemSize(), MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Combine the received subimage into the processed image
            receivedSubImage.copyTo(processedImage(Rect(0, y, width, height)));
        }

        // Display final processed image
        imshow("Processed Image", processedImage);
        waitKey(0); // Wait for a key press to close the window
        cout << "Result Image Displayed" << endl;
    }
    else if (rank < size) {
        int height = imageData.rows; // Height of the whole image
        int stripHeight = height / (size - 1); // Height of each strip
        int startY = (rank - 1) * stripHeight; // Starting Y coordinate for the current rank

        // Process the assigned strip
        Mat subImage = imageData.rowRange(startY, startY + stripHeight);
        Mat processedSubImage = highPassFilter(subImage, kernel);

        // Sending the position and dimensions of the subimage to rank 
        int y = startY; // Initial Y coordinate
        int width = processedSubImage.cols; // Width of each subimage
        height = processedSubImage.rows; // Height of each subimage
        //cout << y;

        MPI_Send(&y, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&width, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&height, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        // Send the subimage data to rank 0
        MPI_Send(processedSubImage.data, width * height * imageData.elemSize(), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string imagePath = "C:/Users/yara/source/repos/hpc-project/lena.png";
    Mat imageData = imread(imagePath, IMREAD_COLOR);

    if (imageData.empty()) {
        cerr << "Error: Could not open or read the image" << endl;
        MPI_Finalize();
        return -1;
    }
    int start_s, stop_s, TotalTime = 0;


    // highpass filter kernel
    Mat kernel = (Mat_<float>(3, 3) <<
        0,-1,0,
        -1,4,-1,
        0,-1,0);
    start_s = clock();

    parallelHighPassFilter(imageData, kernel, rank, size);
    stop_s = clock();
    TotalTime += (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
    std::cout << "time: " << TotalTime << endl;

    MPI_Finalize();
    return 0;
}