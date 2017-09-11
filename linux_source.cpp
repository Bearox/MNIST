#include <iostream>
#include <fstream>
#include <cmath>
#include <set>
#include <cstring>
#include <algorithm>
//#include <opencv2/opencv.hpp>
using namespace std;

//string trainImagePath = "C:\\Users\\t_sund\\Downloads\\train-images-idx3-ubyte\\train-images.idx3-ubyte";
//ifstream fin(filePath, std::ios::binary);
//string trainImagePath = "D:\\work\\TUPUTECH\\dataFile\\train-images.idx3-ubyte";
//string trainLabelPath = "D:\\work\\TUPUTECH\\dataFile\\train-labels.idx1-ubyte";
//string testImagePath = "D:\\work\\TUPUTECH\\dataFile\\t10k-images.idx3-ubyte";
//string testLabelPath = "D:\\work\\TUPUTECH\\dataFile\\t10k-labels.idx1-ubyte";
char trainImagePath[] = "data/train-images.idx3-ubyte";
char trainLabelPath[] = "data/train-labels.idx1-ubyte";
char testImagePath[] = "data/t10k-images.idx3-ubyte";
char testLabelPath[] = "data/t10k-labels.idx1-ubyte";
const int NROW = 28, NCOL = 28;
const int TRAINIMAGE_NUM = 60000;
const int TESTIMAGE_NUM = 10000;

struct Image
{
	int pixel[NROW][NCOL];
	int digit;
}trainIma[TRAINIMAGE_NUM], testIma[TESTIMAGE_NUM];

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
	double *grad_biases[2];
	double **grad_weights[2];
	double *total_grad_biases[2];
	double **total_grad_weights[2];
	//double  *biases[2];
	//double  *weights[2];
	double *input;
	double *mid_value;
	double *sigMidValue;
	double output[10];
	double sigOut[10];
	void updateParameters(int * index, int pos, Image *pIma);
	inline double sShapeFunction(double x);
	inline double updateParametersFunction(double x, double delta);
	void update_grade(Image &ima);
	int test(Image &ima);
public:
	NeuralNetwork();
	~NeuralNetwork();
	void init();
	void train(Image *pIma, Image *tIma);
	int test(Image *pIma);
};

NeuralNetwork::NeuralNetwork()
{
	biases[0] = new double[neural_num];
	biases[1] = new double[output_num];
	grad_biases[0] = new double[neural_num];
	grad_biases[1] = new double[output_num];
	total_grad_biases[0] = new double[neural_num];
	total_grad_biases[1] = new double[output_num];

	weights[0] = new double* [input_num];
	grad_weights[0] = new double*[input_num];
	total_grad_weights[0] = new double*[input_num];
	for (int i = 0; i < input_num; ++i)
	{
		weights[0][i] = new double[neural_num];
		grad_weights[0][i] = new double[neural_num];
		total_grad_weights[0][i] = new double[neural_num];
	}
	weights[1] = new double *[neural_num];
	grad_weights[1] = new double *[neural_num];
	total_grad_weights[1] = new double *[neural_num];
	for (int i = 0; i < neural_num; ++i)
	{
		weights[1][i] = new double[output_num];
		grad_weights[1][i] = new double[output_num];
		total_grad_weights[1][i] = new double[output_num];
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
	delete[] grad_biases[0];
	delete[] grad_biases[1];
	delete[] total_grad_biases[0];
	delete[] total_grad_biases[1];

	for (int i = 0; i < input_num; ++i)
	{
		delete[] weights[0][i];
		delete[] grad_weights[0][i];
		delete[] total_grad_weights[0][i];
	}
	for (int i = 0; i < neural_num; ++i)
	{
		delete[] weights[1][i];
		delete[] grad_weights[1][i];
		delete[] total_grad_weights[1][i];
	}
	delete weights[0];
	delete grad_weights[0];
	delete total_grad_weights[0];
	delete weights[1];
	delete grad_weights[1];
	delete total_grad_weights[1];

	delete[] input;
	delete[] mid_value;
	delete[] sigMidValue;
}

void NeuralNetwork::init()
{
	double sampleNormal();
	for (int i = 0; i < neural_num; ++i)
	{
		biases[0][i] = sampleNormal();
		//cout << biases[0][i] << " ";
	}
	//cout << endl;
	for (int i = 0; i < output_num; ++i)
	{
		biases[1][i] = sampleNormal();
	}

	for (int i = 0; i < input_num; ++i)
	{
		for (int j = 0; j < neural_num; ++j)
		{
			weights[0][i][j] = sampleNormal(); // / sqrt(double(input_num));
		}
	}
	for (int i = 0; i < neural_num; ++i)
	{
		for (int j = 0; j < output_num; ++j)
		{
			weights[1][i][j] = sampleNormal(); // / sqrt(double(input_num));
		}
	}

}

inline double NeuralNetwork::sShapeFunction(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

inline double NeuralNetwork::updateParametersFunction(double x, double delta)
{
	return (x - (eta * (delta / (double)sample_num)));
}

void NeuralNetwork::update_grade(Image &ima)
{
	double realOut[10];
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
		//cout << mid_value[i] << " ";
	}
	//cout << endl;
	for (int i = 0; i < output_num; ++i)
	{
		output[i] = 0.0;
		realOut[i] = 0.0;
		for (int j = 0; j < neural_num; ++j)
		{
			output[i] = output[i] + (sigMidValue[j] * weights[1][j][i]);
		}
		output[i] += biases[1][i];
		sigOut[i] = sShapeFunction(output[i]);
		//cout << sigOut[i] << " " << output[i] << endl;
	}
	
	realOut[ima.digit] = 1.0;

	for (int i = 0; i < 10; ++i)
	{
		grad_biases[1][i] = (sigOut[i] - realOut[i]) * (sigOut[i] * (1 - sigOut[i]));
		//cout << grad_biases[1][i] << " ";
	}
	//cout << endl;

	for (int i = 0; i < neural_num; ++i)
	{	
		for (int j = 0; j < 10; ++j)
		{
			grad_weights[1][i][j] = grad_biases[1][j] * sigMidValue[i];
		}
	}

	for (int i = 0; i < neural_num; ++i)
	{
		double tmpvalue = sigMidValue[i] * (1.0 - sigMidValue[i]);
		grad_biases[0][i] = 0.0;
		for (int j = 0; j < 10; ++j)
		{
			grad_biases[0][i] += (weights[1][i][j] * grad_biases[1][j]);
		}
		grad_biases[0][i] *= tmpvalue;
		for (int j = 0; j < input_num; ++j)
		{
			grad_weights[0][j][i] = grad_biases[0][i] * input[j];
		}
	}
}

void NeuralNetwork::updateParameters(int * index, int pos, Image *pIma)
{
	//double cost = 0.0, grad_cost = 0.0;
	for (int i = 0; i < neural_num; ++i)
	{
		total_grad_biases[0][i] = 0.0;
	}
	for (int i = 0; i < output_num; ++i)
	{
		total_grad_biases[1][i] = 0.0;
	}
	for (int i = 0; i < input_num; ++i)
	{
		for (int j = 0; j < neural_num; ++j)
		{
			total_grad_weights[0][i][j] = 0.0; 
		}
	}
	for (int i = 0; i < neural_num; ++i)
	{
		for (int j = 0; j < output_num; ++j)
		{
			total_grad_weights[1][i][j] = 0.0; 
		}
	}

	for (int ii = 0; ii < sample_num; ++ii)
	{
		update_grade(pIma[index[pos * sample_num + ii]]);
		for (int i = 0; i < neural_num; ++i)
		{
			total_grad_biases[0][i] += grad_biases[0][i];
		}
		for (int i = 0; i < output_num; ++i)
		{
			total_grad_biases[1][i] += grad_biases[1][i];
		}
		for (int i = 0; i < input_num; ++i)
		{
			for (int j = 0; j < neural_num; ++j)
			{
				total_grad_weights[0][i][j] += grad_weights[0][i][j];
			}
		}
		for (int i = 0; i < neural_num; ++i)
		{
			for (int j = 0; j < output_num; ++j)
			{
				total_grad_weights[1][i][j] += grad_weights[1][i][j];
			}
		}
	}
	
	for (int i = 0; i < neural_num; ++i)
	{
		//cout << total_grad_biases[0][i] << " ";
		//cout << biases[0][i] << " ";
		biases[0][i] = updateParametersFunction(biases[0][i], total_grad_biases[0][i]);
		//cout << biases[0][i] << " ";
	}
	//cout << endl;
	for (int i = 0; i < output_num; ++i)
	{
		biases[1][i] = updateParametersFunction(biases[1][i], total_grad_biases[1][i]);
	}
	for (int i = 0; i < input_num; ++i)
	{
		for (int j = 0; j < neural_num; ++j)
		{
			weights[0][i][j] = updateParametersFunction(weights[0][i][j], total_grad_weights[0][i][j]);
		}
	}
	for (int i = 0; i < neural_num; ++i)
	{
		for (int j = 0; j < output_num; ++j)
		{
			weights[1][i][j] = updateParametersFunction(weights[1][i][j], total_grad_weights[1][i][j]);
		}
	}
}

void NeuralNetwork::train(Image *pIma, Image *tIma)
{
	int index[TRAINIMAGE_NUM];
	for (int i = 0; i < TRAINIMAGE_NUM; ++i)
	{
		index[i] = i;
	}
	while (iter_num != 0)
	{
		--iter_num;
		for (int i = 0; i < TRAINIMAGE_NUM; ++i)
		{
			int tmp = rand() % TRAINIMAGE_NUM;
			index[i] = index[i] ^ index[tmp];
			index[tmp] = index[i] ^ index[tmp];
			index[i] = index[i] ^ index[tmp];
		}

		for (int i = 0; i < TRAINIMAGE_NUM / sample_num; ++i)
		{
			updateParameters(index, i, pIma);
			//cout << i << endl;
		}
		int result = test(tIma);
		cout << "Round " << (100 - iter_num) << ": ";
		cout << /*(double)result / (double)TRAINIMAGE_NUM*/ result << endl;
	}
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
			++ans;
	}
	if (ans > 0)
		return ans;
	else
		return -1;
}

double sampleNormal()
{
	double u = ((double)rand() / (RAND_MAX)) * 2 - 1;
	double v = ((double)rand() / (RAND_MAX)) * 2 - 1;
	double r = u * u + v * v;
	if (r == 0 || r > 1)
	{
		return sampleNormal();
	}
	double c = sqrt(-2 * log(r) / r);
	return u * c;
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

	//--------------Read train Image information---------------------------
	ifstream ImageFin(trainImagePath, std::ios::binary);
	ifstream LabelFin(trainLabelPath, std::ios::binary);
	readData(ImageFin, magic_number);
	readData(ImageFin, images_number);
	cout << "Magic Number in Train_set: " << magic_number << endl;
	cout << "The number of images: " << images_number << endl;

	readData(ImageFin, image_rows);
	readData(ImageFin, image_cols);
	cout << "Image Rows: " << image_rows << endl;
	cout << "Image Cols: " << image_cols << endl;

	readImageData(ImageFin, LabelFin, images_number, trainIma);
	ImageFin.close();
	LabelFin.close();

	//--------------Read Test Image information--------------------------------
	ImageFin.open(testImagePath, std::ios::binary);
	LabelFin.open(testLabelPath, std::ios::binary);
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
	nnt.train(trainIma, testIma);
	//nnt.test(testIma);
	return 0;
}