#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

typedef struct Edge {
    int v;
    long long w;
    struct Edge* next;
} Edge;

typedef struct {
    int V;
    int E;
    Edge** adjList;
    int S;
} InputData;

InputData readInput() {
    InputData data;
    scanf("%d %d", &data.V, &data.E);

    data.adjList = (Edge**)malloc((data.V + 1) * sizeof(Edge*));
    for (int i = 0; i <= data.V; ++i) {
        data.adjList[i] = NULL;
    }

    for (int i = 0; i < data.E; ++i) {
        int u, v;
        long long w;
        scanf("%d %d %lld", &u, &v, &w);

        Edge* newEdge = (Edge*)malloc(sizeof(Edge));
        newEdge->v = v;
        newEdge->w = w;
        newEdge->next = data.adjList[u];
        data.adjList[u] = newEdge;
    }

    scanf("%d", &data.S);

    return data;
}

typedef struct HeapNode {
    int v;
    long long dist;
} HeapNode;

typedef struct MinHeap {
    int size;
    int capacity;
    int* pos;
    HeapNode* array;
} MinHeap;

HeapNode newHeapNode(int v, long long dist) {
    HeapNode node;
    node.v = v;
    node.dist = dist;
    return node;
}

MinHeap* createMinHeap(int capacity) {
    MinHeap* heap = (MinHeap*)malloc(sizeof(MinHeap));
    heap->capacity = capacity;
    heap->size = 0;
    heap->pos = (int*)malloc((capacity + 1) * sizeof(int));
    heap->array = (HeapNode*)malloc(capacity * sizeof(HeapNode));
    return heap;
}

void swapHeapNode(HeapNode* a, HeapNode* b) {
    HeapNode temp = *a;
    *a = *b;
    *b = temp;
}

void minHeapify(MinHeap* heap, int idx) {
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;

    if (left < heap->size && heap->array[left].dist < heap->array[smallest].dist)
        smallest = left;

    if (right < heap->size && heap->array[right].dist < heap->array[smallest].dist)
        smallest = right;

    if (smallest != idx) {
        HeapNode smallestNode = heap->array[smallest];
        HeapNode idxNode = heap->array[idx];

        heap->pos[smallestNode.v] = idx;
        heap->pos[idxNode.v] = smallest;

        swapHeapNode(&heap->array[smallest], &heap->array[idx]);

        minHeapify(heap, smallest);
    }
}

int isEmpty(MinHeap* heap) {
    return heap->size == 0;
}

HeapNode extractMin(MinHeap* heap) {
    if (isEmpty(heap))
        return (HeapNode){-1, -1}; 

    HeapNode root = heap->array[0];
    HeapNode lastNode = heap->array[heap->size - 1];
    heap->array[0] = lastNode;

    heap->pos[root.v] = -1;
    heap->pos[lastNode.v] = 0;

    heap->size--;
    minHeapify(heap, 0);

    return root;
}

void decreaseKey(MinHeap* heap, int v, long long dist) {
    int i = heap->pos[v];
    heap->array[i].dist = dist;

    while (i > 0 && heap->array[i].dist < heap->array[(i - 1) / 2].dist) {
        heap->pos[heap->array[i].v] = (i - 1) / 2;
        heap->pos[heap->array[(i - 1) / 2].v] = i;
        swapHeapNode(&heap->array[i], &heap->array[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

int isInHeap(MinHeap* heap, int v) {
    if (heap->pos[v] == -1)
        return 0;
    return 1;
}

void dijkstra(int V, Edge** adjList, int S, long long* dist) {
    MinHeap* minHeap = createMinHeap(V);

    minHeap->size = V;
    for (int v = 1; v <= V; ++v) {
        minHeap->array[v - 1].v = v;
        minHeap->array[v - 1].dist = dist[v];
        minHeap->pos[v] = v - 1;
    }

    dist[S] = 0;
    decreaseKey(minHeap, S, 0);

    while (!isEmpty(minHeap)) {
        HeapNode minNode = extractMin(minHeap);
        int u = minNode.v;
        
        if (minNode.dist == LLONG_MAX) {
            break;
        }

        Edge* currentEdge = adjList[u];
        while (currentEdge != NULL) {
            int v = currentEdge->v;
            long long w = currentEdge->w;

            if (dist[u] != LLONG_MAX && isInHeap(minHeap, v) && (dist[u] + w < dist[v])) {
                dist[v] = dist[u] + w;
                decreaseKey(minHeap, v, dist[v]);
            }

            currentEdge = currentEdge->next;
        }
    }

    free(minHeap->pos);
    free(minHeap->array);
    free(minHeap);
}
int main() {
    InputData input = readInput();

    long long* dist = (long long*)malloc((input.V + 1) * sizeof(long long));
    for (int i = 0; i <= input.V; ++i) {
        dist[i] = LLONG_MAX;
    }

    dijkstra(input.V, input.adjList, input.S, dist);

    for (int i = 1; i <= input.V; ++i) {
        if (dist[i] == LLONG_MAX) {
            printf("-1");
        } else {
            printf("%lld", dist[i]);
        }
        if (i < input.V) printf(" ");
    }
    printf("\n");

    for (int i = 0; i <= input.V; ++i) {
        Edge* current = input.adjList[i];
        while (current != NULL) {
            Edge* temp = current;
            current = current->next;
            free(temp);
        }
    }
    free(input.adjList);
    free(dist);

    return 0;
}