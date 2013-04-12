#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include<xtiffio.h>
#include<geotiff.h>
#include<tiffio.h>
#include<math_functions.h>
#include<cuda_runtime.h>
#include<sys/time.h>

void edge_gpu(unsigned char *buff , unsigned char *buffer_out , int w , int h);
void checkCUDAError(const char* msg);

__global__ void edge_gpu(unsigned char* buff , unsigned char* buffer_out , int w , int h)
{
  int x = blockIdx.x * blockDim.x +threadIdx.x ;
	int y = blockIdx.y * blockDim.y +threadIdx.y; 
	int width = w , height = h ;
	
	if((x>=0 && x < width) && (y>=0 && y<height))
	{
		int hx = -buff[width*(y-1) + (x-1)] + buff[width*(y-1)+(x+1)]
			 -2*buff[width*(y)+(x-1)] + 2*buff[width*(y)+(x+1)]
			 -buff[width*(y+1)+(x-1)] + buff[width*(y+1)+(x+1)];

		int vx = buff[width*(y-1)+(x-1)] +2*buff[width*(y-1)+(x+1)] +buff[width*(y-1)+(x+1)]
			 -buff[width*(y+1)+(x-1)] - 2*buff[width*(y+1)+(x)] - buff[width*(y+1)+(x+1)];
 
		hx = hx/5;
		vx = vx/5;

		int val = (int)sqrt((float)(hx) * (float)(hx) + (float)(vx) * (float)(vx));					

		buffer_out[y * width + x] = (unsigned char) val;							
	}
}
//=========	End Of GPU Function	=========

void checkCUDAError(const char* msg) 
{
	cudaError_t err = cudaGetLastError();
  	if (cudaSuccess != err) 
  	{
    		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    		exit(EXIT_FAILURE);
  	}
}

//=========  Main function  =========

int main(int argc , char** argv)
{
	TIFF *input;
	uint16 photo, bps, spp, rps, comp, pconfig;
	uint32 width, height;
	tsize_t strip_size;
	int strip_max;
	unsigned long buffer_size, offset = 0;
	long int result;
	unsigned char *buffer,*buffer_dev,*buffer_out;
	struct timeval gpu_t1,gpu_t2,gpu_tot;
	
	char infile[100]="parzen.tif", outfile[20]="edge.tif";

	printf("\nSimpe Image Edge Detection Demo Using CUDA.");
	printf("\nEnter .tif Image (8-Bit) : ");
	scanf("%s",infile);

	if((input = XTIFFOpen(infile, "r")) == NULL)
	{
		printf("\nCan not open image %s.", infile);
		exit(42);
	}
	else
	{
		printf("\nTIFF Image Opened Successfully.");
	}

//==========  Extract Image properties....!!  ==========

	TIFFGetField(input, TIFFTAG_BITSPERSAMPLE, &bps);
        TIFFGetField(input, TIFFTAG_SAMPLESPERPIXEL, &spp);
        TIFFGetField(input, TIFFTAG_IMAGEWIDTH, &width);   
        TIFFGetField(input, TIFFTAG_IMAGELENGTH, &height); 
        TIFFGetField(input, TIFFTAG_ROWSPERSTRIP, &rps);   
        TIFFGetField(input, TIFFTAG_COMPRESSION,&comp);    
        TIFFGetField(input, TIFFTAG_PHOTOMETRIC,&photo);   
        TIFFGetField(input, TIFFTAG_PLANARCONFIG, &pconfig);

	printf("\nImage Properties Are : ");

	printf("\nImage Width : %d, Image Height : %d .", width, height);
	printf("\nPhotometric : %d.", photo);		
	printf("\nBits Per Pixels : %d.",bps);		
	printf("\nSamples Per Pixel : %d.",spp);	
	printf("\nRows Per Strip : %d.",rps);
	printf("\nCompression : %d.",comp);
	printf("\nPlanerconfig : %d.",pconfig);

//==========

	strip_size = TIFFStripSize(input);
	strip_max = TIFFNumberOfStrips(input);

	buffer_size = strip_max * strip_size;

	printf("\nNumber of Strips : %d.", strip_max);
	printf("\nStrip Size : %d.", strip_size);
	printf("\nNo of pixels : %ld.", buffer_size);
	
	buffer = (unsigned char *)malloc(buffer_size);	
	
	if(!buffer)
	{
		fprintf(stderr, "\nCould not allocate buffer for uncompressed Image.");
		exit(42);
	}	

	printf("\nNumber of Bytes Required : %d",buffer_size);
	offset = 0;
	for(int i=0;i<strip_max;i++)
	{
		if((result = (long int)TIFFReadEncodedStrip(input, i, buffer + offset, strip_size)) == -1)
		{
			fprintf(stderr,"\nReading Error in Input Strip No : %d");
			exit(42);
		}
		offset = offset + result;
	}

	//	CUDA-GPU Code for Edge Detection .....

	cudaMalloc((void**)&buffer_out,buffer_size);
	checkCUDAError("Memory Allocation");
	
	cudaMalloc((void**)&buffer_dev,buffer_size);
	checkCUDAError("Memory Allocation");
	
	dim3 threadsPerBlock(8,8);
	dim3 numBlocks((width)/8,(height)/8);

//==========  create a stream  ==========

	cudaStream_t stream;
  	cudaStreamCreate(&stream);
  	
	gettimeofday(&gpu_t1,NULL);
	
	cudaMemcpy(buffer_dev , buffer , buffer_size , cudaMemcpyHostToDevice);
	checkCUDAError("Memory Copy From Host To Device");
	
	edge_gpu<<< numBlocks , threadsPerBlock , 0 , stream >>>(buffer_dev , buffer_out, width , height);
	checkCUDAError("Kernel");
		
	unsigned char* buf = (unsigned char*) malloc(buffer_size);
	if(!buf)
	{
		fprintf(stderr,"\nCould no allocate buffer...Insufficient Memory.");
	}

	cudaMemcpy(buf , buffer_out , buffer_size  , cudaMemcpyDeviceToHost);
	checkCUDAError("Memory Copy From Device To Host");

	gettimeofday(&gpu_t2,NULL);
	timersub(&gpu_t2,&gpu_t1,&gpu_tot);

        XTIFFClose(input);

	if((input = XTIFFOpen(outfile, "w")) == NULL)
	{
		printf("\nCan not open output file.");
		exit(42);
	}

	// Write Image properties into output file....

        TIFFSetField(input, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(input, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(input, TIFFTAG_BITSPERSAMPLE, bps);
        TIFFSetField(input, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(input, TIFFTAG_ROWSPERSTRIP, height);
        TIFFSetField(input, TIFFTAG_COMPRESSION, comp);
        TIFFSetField(input, TIFFTAG_PHOTOMETRIC, photo);
        TIFFSetField(input, TIFFTAG_PLANARCONFIG, pconfig);	

	TIFFWriteEncodedStrip(input,0, buf, buffer_size);
	
	printf("\n\nTime Required for GPU : ");
	printf(" %d Seconds , %d Milliseconds.",gpu_tot.tv_sec,gpu_tot.tv_usec);

        XTIFFClose(input);

	cudaFree(buffer_dev);
	cudaFree(buffer_out);
	cudaStreamDestroy(stream);	

	free(buf);
	
	printf("\n\nDONE...!!...Copy Successful...!!\n\n");

	return 1;
}

