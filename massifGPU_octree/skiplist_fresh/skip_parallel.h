#include <limits.h>
typedef struct Node Node;
typedef struct Skiplist Skiplist;
#define MIN_VAL INT_MIN

struct OctreeNode {
  int val;
   
};

typedef struct OctreeNode E;

Skiplist *skiplist_create(void);
void skiplist_destroy(Skiplist *sl);
__device__ void skiplist_insert(Skiplist *sl, E elem);
__device__ void skiplist_remove(Skiplist *sl, E elem);
int skiplist_size(Skiplist *sl);
E *skiplist_gather(Skiplist *sl, int *dim);

/* for traversal */
__device__ Node *skiplist_head(Skiplist *sl);
__device__ Node *node_next(Node *node);

// error-checking macro
#define CHECK(cuda) \
  _check( (cuda) , __FILE__, __LINE__)

void _check(cudaError_t cs, const char *file, long line);
