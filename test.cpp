#include <iostream>
#include <fstream>
#include <cmath>
#include <set>
#include <cstring>
#include <algorithm>
//#include <opencv2/opencv.hpp>
using namespace std;

char testImagePath[] = "data/t10k-images.idx3-ubyte";
char testLabelPath[] = "data/t10k-labels.idx1-ubyte";
char modelFilePath[] = "model.txt";
const int NROW = 28, NCOL = 28;
const int TESTIMAGE_NUM = 10000;

struct Image
{
	int pixel[NROW][NCOL];
	int digit;
}testIma[TESTIMAGE_NUM];

class NeuralNetwork
{
	const int input_num = NROW * NCOL;
	const int output_num = 10;
	const int neural_num = 30;
	const int layer_num = 3;
	int iter_num = 100;
	const int sample_num = 10;
	const double eta = 3.0;
	double *biases[2];
	double **weights[2];
	double *input;
	double *mid_value;
	double *sigMidValue;
	double output[10];
	double sigOut[10];
	inline double sShapeFunction(double x);
	int test(Image &ima);
public:
	NeuralNetwork();
	~NeuralNetwork();
	void init();
	int test(Image *pIma);
};

NeuralNetwork::NeuralNetwork()
{
	biases[0] = new double[neural_num];
	biases[1] = new double[output_num];
	
	weights[0] = new double* [input_num];
	
	for (int i = 0; i < input_num; ++i)
	{
		weights[0][i] = new double[neural_num];
		
	}
	weights[1] = new double *[neural_num];
	
	for (int i = 0; i < neural_num; ++i)
	{
		weights[1][i] = new double[output_num];
	}

	input = new double[input_num];
	mid_value = new double[neural_num];
	sigMidValue = new double[neural_num];
	init();
}

NeuralNetwork::~NeuralNetwork()
{
	delete[] biases[0];
	delete[] biases[1];

	for (int i = 0; i < input_num; ++i)
	{
		delete[] weights[0][i];	
	}
	for (int i = 0; i < neural_num; ++i)
	{
		delete[] weights[1][i];	
	}
	delete weights[0];
	delete weights[1];
	

	delete[] input;
	delete[] mid_value;
	delete[] sigMidValue;
}

void NeuralNetwork::init()
{
    ifstream modelIn(modelFilePath);
	for (int i = 0; i < neural_num; ++i)
	{
		modelIn >> biases[0][i];
	}
	
	for (int i = 0; i < output_num; ++i)
	{
		modelIn >> biases[1][i];
	}

	for (int i = 0; i < input_num; ++i)
	{
		for (int j = 0; j < neural_num; ++j)
		{
			modelIn >> weights[0][i][j];
		}
	}
	for (int i = 0; i < neural_num; ++i)
	{
		for (int j = 0; j < output_num; ++j)
		{
			modelIn >> weights[1][i][j]; 
		}
	}
    modelIn.close();
}

inline double NeuralNetwork::sShapeFunction(double x)
{
	return 1.0 / (1.0 + exp(-x));
}


int NeuralNetwork::test(Image &ima)
{
	for (int i = 0; i < NROW; ++i)
	{
		for (int j = 0; j < NCOL; ++j)
		{
			input[i * NROW + j] = (double)ima.pixel[i][j] / 255.0;
		}
	}
	for (int i = 0; i < neural_num; ++i)
	{
		mid_value[i] = 0.0;
		for (int j = 0; j < input_num; ++j)
		{
			mid_value[i] = mid_value[i] + (input[j] * weights[0][j][i]);
		}
		mid_value[i] += biases[0][i];
		sigMidValue[i] = sShapeFunction(mid_value[i]);
	}
	for (int i = 0; i < output_num; ++i)
	{
		output[i] = 0.0;
		for (int j = 0; j < neural_num; ++j)
		{
			output[i] = output[i] + (sigMidValue[j] * weights[1][j][i]);
		}
		output[i] += biases[1][i];
		sigOut[i] = sShapeFunction(output[i]);
	}
	int max_pos = 0;
	double max_value = sigOut[0];
	for (int i = 1; i < output_num; ++i)
	{
		if (sigOut[i] > max_value)
		{
			max_pos = i;
			max_value = sigOut[i];
		}
	}
	if (max_pos == ima.digit)
		return 1;
	else
		return 0;
}

int NeuralNetwork::test(Image *pIma)
{
	int ans = 0;
	for (int i = 0; i < TESTIMAGE_NUM; ++i)
	{
		if (test(pIma[i]))
        {
            //cout << "Yes!" << endl;
			++ans;
        }
        else
        {
            cout << "Pitty! The " << i << "th picture is not recognized!" << endl;
        }
	}
	cout << "Total " << ans << " pictures is recognized!" << endl;
}

void readData(ifstream &fin, int &value)
{
	fin.read((char*)&value, sizeof(int));
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = value & 255;
	ch2 = (value >> 8) & 255;
	ch3 = (value >> 16) & 255;
	ch4 = (value >> 24) & 255;
	value = ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void readImageData(ifstream &imageIn, ifstream &labelIn, int imageNum, Image *pIma)
{
	//cv::Mat outImage = cv::Mat::zeros(NROW, NCOL, CV_8UC1);
	int magic_num, label_num;
	unsigned char tmp;
	readData(labelIn, magic_num);  //magic_num in label file
	readData(labelIn, label_num);  //image_num in label file
	cout << "Magic Number in Label File: " << magic_num << endl;
	cout << "The number of labels: " << label_num << endl;
	for (int i = 0; i < imageNum; ++i)
	{
		//cout << i << endl;
		for (int pRow = 0; pRow < NROW; ++pRow)
		{
			for (int pCol = 0; pCol < NCOL; ++pCol)
			{
				imageIn.read((char*)&tmp, sizeof(tmp));
				pIma[i].pixel[pRow][pCol] = (int)tmp;

				//outImage.at<uchar>(pRow, pCol) = pIma[i].pixel[pRow][pCol];
			}
		}
		labelIn.read((char*)&tmp, sizeof(tmp));
		pIma[i].digit = (int)tmp;

		//cv::imshow("Image-show", outImage);
		//cv::waitKey(0);
		//cout <<  endl << endl << trainIma[i].digit << endl << endl << endl;
	}
	//cv::imwrite("D:\\sample.jpg", outImage);
}

int main()
{

	int magic_number, images_number;
	int image_rows, image_cols;

	//--------------Read Test Image information--------------------------------
	ifstream ImageFin(testImagePath, std::ios::binary);
	ifstream LabelFin(testLabelPath, std::ios::binary);
	readData(ImageFin, magic_number);
	readData(ImageFin, images_number);
	cout << "Magic Number in Test_set: " << magic_number << endl;
	cout << "The number of images: " << images_number << endl;

	readData(ImageFin, image_rows);
	readData(ImageFin, image_cols);
	cout << "Image Rows: " << image_rows << endl;
	cout << "Image Cols: " << image_cols << endl;

	readImageData(ImageFin, LabelFin, images_number, testIma);
	ImageFin.close();

	//-------------------------------------------------------------------------
	NeuralNetwork nnt;
	//nnt.train(trainIma, testIma);
	nnt.test(testIma);
	return 0;
}