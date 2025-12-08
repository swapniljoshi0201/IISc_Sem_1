#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int op;
    int n;
    int* arr;
    int* idx;
    int* key;
} InputData;

int toInt(const char* s) {
    int val = 0;
    int sign = 1;
    int i = 0;

    if (s[0] == '-') {
        sign = -1;
        i++;
    }
    
    for (; s[i] != '\0'; i++) {
        val = val * 10 + (s[i] - '0');
    }

    return sign * val;
}

InputData readInput() {
    InputData data;
    scanf("%d %d", &data.op, &data.n);

    if (data.n > 0) {
        data.arr = (int*)malloc(data.n * sizeof(int));
        for (int i = 0; i < data.n; i++) {
            scanf("%d", &data.arr[i]);
        }
    } else {
        data.arr = NULL;
        int c;
        while ((c = getchar()) != '\n' && c != EOF);
    }

    char idx_str[64], key_str[64];
    scanf("%s %s", idx_str, key_str);

    if (strcmp(idx_str, "None") != 0) {
        int val = toInt(idx_str);
        data.idx = (int*)malloc(sizeof(int));
        *data.idx = val;
    } else {
        data.idx = NULL;
    }

    if (strcmp(key_str, "None") != 0) {
        int val = toInt(key_str);
        data.key = (int*)malloc(sizeof(int));
        *data.key = val;
    } else {
        data.key = NULL;
    }

    return data;
}

void swap(int *a, int *b) { 
    int t = *a; 
    *a = *b; 
    *b = t; 
    
}

void bubbleUp(int *arr, int i) {
    while (i > 0 && arr[(i-1)/2] > arr[i]) {
        swap(&arr[i], &arr[(i-1)/2]);
        i = (i-1)/2;
    }
}

void heapify(int *arr, int n, int i) {
    int smallest = i;
    int l = 2*i+1;
    int r = 2*i+2;
    if (l < n && arr[l] < arr[smallest]) smallest = l;
    if (r < n && arr[r] < arr[smallest]) smallest = r;
    if (smallest != i) { 
        swap(&arr[i], &arr[smallest]); heapify(arr,n,smallest); 
        
    }
}

void insert(int **arr_ptr, int *n, int val) {
    *arr_ptr = (int*)realloc(*arr_ptr, (*n+1)*sizeof(int));
    (*arr_ptr)[*n] = val; 
    (*n)++;
    bubbleUp(*arr_ptr, *n-1);
}

int extractMin(int *arr, int *n) {
    if (*n <= 0) return -1;
    if (*n == 1) { 
        (*n)--; 
        return arr[0]; 
    }
    int root = arr[0];
    arr[0] = arr[*n-1]; 
    (*n)--;
    heapify(arr, *n, 0);
    return root;
}

void decreaseKey(int *arr, int n, int idx, int val) {
    if (idx >= n) return;
    arr[idx] = val; 
    bubbleUp(arr, idx);
}

void performHeapify(int *arr, int n, int idx) { 
    heapify(arr,n,idx); 
    
}

void heapSort(int *arr, int n) {
    int *heap = NULL, size = 0;
    for (int i = 0; i < n; i++) 
        insert(&heap, &size, arr[i]);
    for (int i = 0; i < n; i++) 
        arr[i] = extractMin(heap,&size);
    free(heap);
}

void printArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d", arr[i]);
        if (i < n-1) printf(" ");
    }
    printf("\n");
}


int main() {
    InputData input = readInput();

    int op = input.op;
    int n = input.n;
    int* arr = input.arr;
    int* idx = input.idx;
    int* key = input.key;

      int *heap_arr = NULL, heap_size = 0;
    for (int i = 0; i < n; i++) 
    insert(&heap_arr, &heap_size, arr[i]);
    switch(op) {
        case 1: insert(&heap_arr,&heap_size,*key); 
        printArray(heap_arr,heap_size); 
        break;
        case 2: printf("%d\n",extractMin(heap_arr,&heap_size)); 
        break;
        case 3: decreaseKey(heap_arr,heap_size,*idx,*key); 
        printArray(heap_arr,heap_size); 
        break;
        case 4: performHeapify(heap_arr,heap_size,*idx); 
        printArray(heap_arr,heap_size); 
        break;
        case 5: heapSort(arr,n); 
        printArray(arr,n); 
        break;
    }
    
  	// Remember to free allocated memory
    if (arr) free(arr);
    if (idx) free(idx);
    if (key) free(key);
    if (heap_arr) free(heap_arr);

    return 0;
}