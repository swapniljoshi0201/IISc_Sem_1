#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

typedef struct {
    int C;          // target amount
    int n;          // number of denominations
    int* Denom;     // array of denominations
} InputData;

InputData readInput() {
    InputData data;
    scanf("%d %d", &data.C, &data.n);

    data.Denom = (int*)malloc(data.n * sizeof(int));

    for (int i = 0; i < data.n; i++) {
        scanf("%d", &data.Denom[i]);
    }

    return data;
}

int minCoins(int C, int* Denom, int n) {
    
  int* minCount = (int*)malloc((C + 1) * sizeof(int));

    minCount[0] = 0;

    for (int i = 1; i <= C; i++) {
        minCount[i] = INT_MAX;
    }

    for (int i = 1; i <= C; i++) {
        for (int j = 0; j < n; j++) {
            
            int coin = Denom[j];
            
            if (coin <= i) {
                
                if (minCount[i - coin] != INT_MAX) {
                    
                    int currentMin = minCount[i - coin] + 1;
                    
                    if (currentMin < minCount[i]) {
                        minCount[i] = currentMin;
                    }
                }
            }
        }
    }

    int result = minCount[C];
    
    free(minCount);
    
    return result;
}

int main() {
    InputData input = readInput();

    /*
     * Call the minCoins function and output the result.
     * The output should be a single integer representing the minimum
     * number of coins needed to make the amount C.
     */
    int result = minCoins(input.C, input.Denom, input.n);
    printf("%d\n", result);

    // Remember to free allocated memory
    free(input.Denom);

    return 0;
}