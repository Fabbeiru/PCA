#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <stdlib.h>
#include <mkl.h>
#include <omp.h>

using namespace std;

vector<string> readCSV(string fname) {
	vector<string> row;
	string line, word;

	fstream file(fname, ios::in);
	if (file.is_open())
	{
		while (getline(file, line))
		{
			stringstream str(line);

			while (getline(str, word, ',')) row.push_back(word);
		}
	}
	else cout << "Could not open the file\n";

	row[0] = "0.72694";

	return row;
}

double* vectorToDouble(vector<string> rawData, double* data) {
	#pragma omp parallel num_threads(8)
	{
		#pragma omp for nowait
		for (int i = 0; i < rawData.size(); i++) {
			data[i] = stod(rawData[i]);
		}
	}
		
	return data;
}

void centerData(double* data, int m, int n) {
	double* media = new double[n];

	#pragma omp parallel num_threads(7)
	{
		#pragma omp sections nowait
		{
			#pragma omp section
			{
				double total = 0;
				for (int i = 0; i < m; i++) {
					total += data[i * n];
				}
				media[0] = total / m;
				for (int i = 0; i < m; i++) {
					data[i * n] -= media[0];
				}
			}
			#pragma omp section
			{
				double total = 0;
				for (int i = 0; i < m; i++) {
					total += data[i * n+1];
				}
				media[1] = total / m;
				for (int i = 0; i < m; i++) {
					data[i * n+1] -= media[1];
				}
			}
			#pragma omp section
			{
				double total = 0;
				for (int i = 0; i < m; i++) {
					total += data[i * n+2];
				}
				media[2] = total / m;
				for (int i = 0; i < m; i++) {
					data[i * n+2] -= media[2];
				}
			}
			#pragma omp section
			{
				double total = 0;
				for (int i = 0; i < m; i++) {
					total += data[i * n+3];
				}
				media[3] = total / m;
				for (int i = 0; i < m; i++) {
					data[i * n+3] -= media[3];
				}
			}
			#pragma omp section
			{
				double total = 0;
				for (int i = 0; i < m; i++) {
					total += data[i * n+4];
				}
				media[4] = total / m;
				for (int i = 0; i < m; i++) {
					data[i * n+4] -= media[4];
				}
			}
			#pragma omp section
			{
				double total = 0;
				for (int i = 0; i < m; i++) {
					total += data[i * n+5];
				}
				media[5] = total / m;
				for (int i = 0; i < m; i++) {
					data[i * n+5] -= media[5];
				}
			}
			#pragma omp section
			{
				double total = 0;
				for (int i = 0; i < m; i++) {
					total += data[i * n+6];
				}
				media[6] = total / m;
				for (int i = 0; i < m; i++) {
					data[i * n+6] -= media[6];
				}
			}
		}
	}
}

double* getCovarianza(double* XC, int rows, int cols) {
	MKL_INT m, n, k, alpha, beta, lda, ldb, ldc;
	m = cols;
	n = cols;
	k = rows;
	alpha = 1;
	beta = 0;
	lda = m;
	ldb = n;
	ldc = m;
	double* XC2 = new double[2380];
	memcpy(XC2, XC, sizeof(double)*2380);
	double* Z = new double[49]{0};

	cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, k, alpha, XC2, lda, XC, ldb, beta, Z, ldc);

	#pragma omp parallel num_threads(8)
	{
		#pragma omp for nowait
		for (int i = 0; i < 49; i++) {
			Z[i] /= rows;
		}
	}
	
	return Z;
}

void getEigenValuesAndVectors(double* eigenValues, double* eigenVectors) {
	char jobz = 'V';
	char uplo = 'U';
	lapack_int lda = 7;

	LAPACKE_dsyev(LAPACK_ROW_MAJOR, jobz, uplo, 7, eigenVectors, lda, eigenValues);
}

void sortValues(double* eigenValue) {
	for (int i = 7; i >= 0; i--)
		for (int j = 7; j > 7 - i; j--)
			if (eigenValue[j] > eigenValue[j - 1])
				swap(eigenValue[j], eigenValue[j - 1]);
}

void sortVectors(double* eigenVectors) {
	int cont = 1;
	double* aux = new double[49]{ 0 };
	memcpy(aux, eigenVectors, sizeof(double) * 49);

	for (int j = 7 - 1; j >= 0; j--) {
		for (int i = 1; i <= 7; i++) {
			eigenVectors[(7 * (i - 1)) + (cont - 1)] = aux[(i * 7) - cont];
		}
		cont++;
	}
}

double* getPCA(double* dataAux, double* eigenVector, int rows, int cols) {
	MKL_INT m, n, k, alpha, beta, lda, ldb, ldc;
	m = rows;
	n = cols;
	k = cols;
	alpha = 1;
	beta = 0;
	lda = n;
	ldb = n;
	ldc = n;
	double* res = new double[2380]{ 0 };

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, dataAux, lda, eigenVector, ldb, beta, res, ldc);

	return res;
}

int main()
{
	// Paso 1 -> Obtener datos
	int m = 340;
	int n = 7;
	string fname = "leaf_modificado.csv";
	vector<string> rawData = readCSV(fname);
	double* data = new double[2380];
	data = vectorToDouble(rawData, data);

	// Paso 2 -> Centrar datos
	centerData(data, m, n);
	double* dataAux = new double[2380];
	memcpy(dataAux, data, sizeof(double) * 2380);

	// Paso 3 -> Obtener matriz de covarianza y autovalores y autovectores
	double* Z = new double[49];
	Z = getCovarianza(data, m, n);

	double* eigenValues = new double[7]{ 0 };
	double* eigenVectors = new double[49]{ 0 };
	memcpy(eigenVectors, Z, sizeof(double) * 49);
	getEigenValuesAndVectors(eigenValues, eigenVectors);
	sortValues(eigenValues);
	sortVectors(eigenVectors);

	// Paso 4 -> Obtener valores finales
	double* pca = new double[2380]{ 0 };
	pca = getPCA(dataAux, eigenVectors, m, n);

	cout << "\n\tAutovalores:" << "\n\t";
	for (int i = 0; i < 7; i++) {
		cout << eigenValues[i] << " ";
	}
	cout << "\n\n\tAutovectores:" << "\t";
	for (int i = 0; i < 35; i++) {
		if (i % 7 == 0) cout << "\n\t";
		cout << pca[i] << " ";
	}
	cout << "\n\n";

	return 0;
}