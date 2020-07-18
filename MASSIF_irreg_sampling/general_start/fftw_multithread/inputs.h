#define NX 512
#define NY 512
#define NZ 512
#define K 32 //dimension of small cube
#define startX 4 //start points of domain
#define startY 4
#define startZ 0
#define B 4096 // Number of pencils in one batch
#define NRANK 3
#define DS1 16//downsample rate
#define DS2 16 //downsample rate for z dim
#define DS 16
#define TO_PRINT 0
#define OCTREE_FINEST 16 //has to be greater than or equal to DS rate
#define NTHREADS 14 //FFTW threads based on bridges GPU node number of cores/CPU
