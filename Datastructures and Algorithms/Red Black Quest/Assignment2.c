#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct InputData {
    int op;                 // operation code
    int n;                  // number of elements
    int* arr;               // input array
    int* searchElement;     // element for insert operation if op = 1, search operation if op = 3
    int* deleteElement;     // element for delete operation
    char traversalType[20]; // traversal type string
};

int toInt(const char* s) {
    int val = 0, sign = 1, i = 0;
    if (s[0] == '-') {
        sign = -1;
        i++;
    }
    for (; s[i] != '\0'; i++) {
        val = val * 10 + (s[i] - '0');
    }
    return sign * val;
}

struct InputData readInput() {
    struct InputData data;
    char srch_str[20], del_str[20];

    scanf("%d %d", &data.op, &data.n);

    data.arr = (int*)malloc(data.n * sizeof(int));
    for (int i = 0; i < data.n; ++i) {
        scanf("%d", &data.arr[i]);
    }

    scanf("%s %s", srch_str, del_str);

    if (strcmp(srch_str, "None") != 0) {
        data.searchElement = (int*)malloc(sizeof(int));
        *data.searchElement = toInt(srch_str);
    } else {
        data.searchElement = NULL;
    }

    if (strcmp(del_str, "None") != 0) {
        data.deleteElement = (int*)malloc(sizeof(int));
        *data.deleteElement = toInt(del_str);
    } else {
        data.deleteElement = NULL;
    }

    if (data.op == 4) {
        scanf("%s", data.traversalType);
    } else {
        data.traversalType[0] = '\0';
    }

    return data;
}

typedef enum { RED = 0, BLACK = 1 } Color;

typedef struct Node {
    int key;
    Color color;
    struct Node *left, *right, *parent;
} Node;

typedef struct RedBlackTree {
    Node *root;
    Node *NIL;
} RedBlackTree;

Node* create_node(RedBlackTree* rbt, int key) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->key = key; node->color = RED;
    node->left = node->right = node->parent = rbt->NIL;
    return node;
}

RedBlackTree* createRedBlackTree() {
    RedBlackTree* rbt = (RedBlackTree*)malloc(sizeof(RedBlackTree));
    rbt->NIL = (Node*)malloc(sizeof(Node));
    rbt->NIL->color = BLACK;
    rbt->NIL->left = rbt->NIL->right = rbt->NIL->parent = rbt->NIL;
    rbt->root = rbt->NIL;
    return rbt;
}

/* Rotation and Fixups */
void leftRotate(RedBlackTree* rbt, Node* x) {
    Node* y = x->right; x->right = y->left;
    if (y->left != rbt->NIL) y->left->parent = x;
    y->parent = x->parent;
    if (x->parent == rbt->NIL) rbt->root = y;
    else if (x == x->parent->left) x->parent->left = y;
    else x->parent->right = y;
    y->left = x; x->parent = y;
}

void rightRotate(RedBlackTree* rbt, Node* x) {
    Node* y = x->left; x->left = y->right;
    if (y->right != rbt->NIL) y->right->parent = x;
    y->parent = x->parent;
    if (x->parent == rbt->NIL) rbt->root = y;
    else if (x == x->parent->right) x->parent->right = y;
    else x->parent->left = y;
    y->right = x; x->parent = y;
}

void insertFixup(RedBlackTree* rbt, Node* z) {
    while (z->parent->color == RED) {
        if (z->parent == z->parent->parent->left) {
            Node* y = z->parent->parent->right;
            if (y->color == RED) {
                z->parent->color = y->color = BLACK;
                z->parent->parent->color = RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->right) { z = z->parent; leftRotate(rbt, z); }
                z->parent->color = BLACK;
                z->parent->parent->color = RED;
                rightRotate(rbt, z->parent->parent);
            }
        } else {
            Node* y = z->parent->parent->left;
            if (y->color == RED) {
                z->parent->color = y->color = BLACK;
                z->parent->parent->color = RED;
                z = z->parent->parent;
            } else {
                if (z == z->parent->left) { z = z->parent; rightRotate(rbt, z); }
                z->parent->color = BLACK;
                z->parent->parent->color = RED;
                leftRotate(rbt, z->parent->parent);
            }
        }
    }
    rbt->root->color = BLACK;
}

void insert(RedBlackTree* rbt, int value) {
    Node* z = create_node(rbt, value);
    Node* y = rbt->NIL; Node* x = rbt->root;
    while (x != rbt->NIL) { y = x; x = (z->key < x->key ? x->left : x->right); }
    z->parent = y;
    if (y == rbt->NIL) rbt->root = z;
    else if (z->key < y->key) y->left = z;
    else y->right = z;
    insertFixup(rbt, z);
}

Node* treeMinimum(RedBlackTree* rbt, Node* x) { while (x->left != rbt->NIL) x = x->left; return x; }
void rbTransplant(RedBlackTree* rbt, Node* u, Node* v) {
    if (u->parent == rbt->NIL) rbt->root = v;
    else if (u == u->parent->left) u->parent->left = v;
    else u->parent->right = v;
    v->parent = u->parent;
}

/* Delete Fixup & Single delete */
void deleteFixup(RedBlackTree* rbt, Node* x) {
    while (x != rbt->root && x->color == BLACK) {
        if (x == x->parent->left) {
            Node* w = x->parent->right;
            if (w->color == RED) { w->color = BLACK; x->parent->color = RED; leftRotate(rbt, x->parent); w = x->parent->right; }
            if (w->left->color == BLACK && w->right->color == BLACK) { w->color = RED; x = x->parent; }
            else {
                if (w->right->color == BLACK) { w->left->color = BLACK; w->color = RED; rightRotate(rbt, w); w = x->parent->right; }
                w->color = x->parent->color; x->parent->color = BLACK; w->right->color = BLACK; leftRotate(rbt, x->parent); x = rbt->root;
            }
        } else {
            Node* w = x->parent->left;
            if (w->color == RED) { w->color = BLACK; x->parent->color = RED; rightRotate(rbt, x->parent); w = x->parent->left; }
            if (w->left->color == BLACK && w->right->color == BLACK) { w->color = RED; x = x->parent; }
            else {
                if (w->left->color == BLACK) { w->right->color = BLACK; w->color = RED; leftRotate(rbt, w); w = x->parent->left; }
                w->color = x->parent->color; x->parent->color = BLACK; w->left->color = BLACK; rightRotate(rbt, x->parent); x = rbt->root;
            }
        }
    }
    x->color = BLACK;
}

int deleteSingle(RedBlackTree* rbt, int key) {
    Node* z = rbt->root;
    while (z != rbt->NIL) { if (key == z->key) break; else if (key < z->key) z = z->left; else z = z->right; }
    if (z == rbt->NIL) return 0;

    Node* y = z; Node* x; Color y_color = y->color;
    if (z->left == rbt->NIL) { x = z->right; rbTransplant(rbt, z, z->right); }
    else if (z->right == rbt->NIL) { x = z->left; rbTransplant(rbt, z, z->left); }
    else {
        y = treeMinimum(rbt, z->right);
        y_color = y->color; x = y->right;
        if (y->parent != z) { rbTransplant(rbt, y, y->right); y->right = z->right; y->right->parent = y; }
        rbTransplant(rbt, z, y); y->left = z->left; y->left->parent = y; y->color = z->color;
    }
    free(z); if (y_color == BLACK) deleteFixup(rbt, x);
    return 1;
}

void deleteValue(RedBlackTree* rbt, int value) { deleteSingle(rbt, value); }

int search(RedBlackTree* rbt, int value) {
    Node* cur = rbt->root;
    while (cur != rbt->NIL) {
        if (value == cur->key) return 1;
        else if (value < cur->key) cur = cur->left;
        else cur = cur->right;
    }
    return 0;
}

/* Traversals */
int inorderHelper(RedBlackTree* rbt, Node* node, int out[], int idx) {
    if (node == rbt->NIL) return idx;
    idx = inorderHelper(rbt, node->left, out, idx);
    out[idx++] = node->key;
    idx = inorderHelper(rbt, node->right, out, idx);
    return idx;
}
int inorderTraversal(RedBlackTree* rbt, int out[]) { return inorderHelper(rbt, rbt->root, out, 0); }

int preorderHelper(RedBlackTree* rbt, Node* node, int out[], int idx) {
    if (node == rbt->NIL) return idx;
    out[idx++] = node->key;
    idx = preorderHelper(rbt, node->left, out, idx);
    idx = preorderHelper(rbt, node->right, out, idx);
    return idx;
}
int preorderTraversal(RedBlackTree* rbt, int out[]) { return preorderHelper(rbt, rbt->root, out, 0); }

int postorderHelper(RedBlackTree* rbt, Node* node, int out[], int idx) {
    if (node == rbt->NIL) return idx;
    idx = postorderHelper(rbt, node->left, out, idx);
    idx = postorderHelper(rbt, node->right, out, idx);
    out[idx++] = node->key;
    return idx;
}
int postorderTraversal(RedBlackTree* rbt, int out[]) { return postorderHelper(rbt, rbt->root, out, 0); }

typedef struct { Node** data; int front, back, cap; } Queue;
Queue* q_create(int cap) { Queue* q = malloc(sizeof(Queue)); q->data = malloc(sizeof(Node*)*cap); q->front = q->back = 0; q->cap = cap; return q; }
void q_push(Queue* q, Node* v) { if(q->back==q->cap){int n=q->cap*2;q->data=realloc(q->data,sizeof(Node*)*n);q->cap=n;} q->data[q->back++]=v; }
Node* q_pop(Queue* q) { return (q->front==q->back)?NULL:q->data[q->front++]; }
int q_empty(Queue* q){ return q->front==q->back; }
void q_free(Queue* q){ free(q->data); free(q); }

int levelOrderTraversal(RedBlackTree* rbt, int out[]) {
    if (rbt->root==rbt->NIL) return 0;
    Queue* q=q_create(128); q_push(q,rbt->root); int idx=0;
    while(!q_empty(q)) {
        Node* cur=q_pop(q); if(cur==rbt->NIL) continue;
        out[idx++]=cur->key;
        if(cur->left!=rbt->NIL) q_push(q,cur->left);
        if(cur->right!=rbt->NIL) q_push(q,cur->right);
    }
    q_free(q); return idx;
}

void free_nodes(RedBlackTree* rbt, Node* node) {
    if(node==rbt->NIL||node==NULL) return;
    free_nodes(rbt,node->left); free_nodes(rbt,node->right); free(node);
}
void freeRedBlackTree(RedBlackTree* rbt) {
    if(!rbt) return; 
    free_nodes(rbt,rbt->root); 
    free(rbt->NIL); 
    free(rbt);
}

int main() {
    struct InputData input = readInput();
    RedBlackTree* rbt = createRedBlackTree();
    for(int i=0;i<input.n;i++) insert(rbt,input.arr[i]);

    if(input.op==1) {
        if(input.searchElement) insert(rbt,*input.searchElement);
        int *out=malloc(sizeof(int)*(input.n+5)); int cnt=inorderTraversal(rbt,out);
        for(int i=0;i<cnt;i++){ if(i)printf(" "); printf("%d",out[i]); } printf("\n"); free(out);
    } else if(input.op==2) {
        if(input.deleteElement) while(search(rbt,*input.deleteElement)) deleteValue(rbt,*input.deleteElement);
        int *out=malloc(sizeof(int)*(input.n+5)); int cnt=inorderTraversal(rbt,out);
        for(int i=0;i<cnt;i++){ if(i)printf(" "); printf("%d",out[i]); } printf("\n"); free(out);
    } else if(input.op==3) {
        int found = (input.searchElement)?search(rbt,*input.searchElement):0;
        printf(found?"True\n":"False\n");
    } else if(input.op==4) {
        int *out=malloc(sizeof(int)*(input.n+5)); int cnt=0;
        if(strcmp(input.traversalType,"inorder")==0) cnt=inorderTraversal(rbt,out);
        else if(strcmp(input.traversalType,"preorder")==0) cnt=preorderTraversal(rbt,out);
        else if(strcmp(input.traversalType,"postorder")==0) cnt=postorderTraversal(rbt,out);
        else if(strcmp(input.traversalType,"levelorder")==0) cnt=levelOrderTraversal(rbt,out);
        for(int i=0;i<cnt;i++){ if(i) printf(" "); printf("%d",out[i]); } printf("\n"); free(out);
    }

    free(input.arr);
    if(input.searchElement) free(input.searchElement);
    if(input.deleteElement) free(input.deleteElement);
    freeRedBlackTree(rbt);
    return 0;
}