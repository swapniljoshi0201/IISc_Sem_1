The Change Maker — Minimum Change with Dynamic Programming
You are given a set of currency denominations and a target amount C. Your task is to determine the minimum number of currency notes or coins required to make change for the given amount.

Each denomination can be used any number of times. You may assume that there is always at least one denomination with value 1, ensuring that making change is always possible.

Input Format

The input consists of:

An integer C, representing the target amount.
An integer n, representing the number of currency denominations.
An array Denom[1..n], containing the values of the available denominations.
Constraints

1 ≤ C ≤ 10⁴
1 ≤ n ≤ 100
1 ≤ Denom[i] ≤ C
Output Format

Output a single integer — the minimum number of currency notes or coins needed to make the amount C.

Sample Input 0

11
3
1 2 5
Sample Output 0

3
Sample Input 1

12
3
1 4 5
Sample Output 1

3
