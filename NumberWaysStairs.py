# To clim n stairs:
# - the person can climb 1 step and will have (n-1) remaining steps to climb
# - the person can climb 2 steps and will have (n-2) remaining steps to climb
# This means NumberWays(n) = NumberWays(n-1) + NumberWays(n-2)

def NumberWays(n):
    res =[0 for i in range(n+1)] 
    res[0], res[1] =1,1
    for i in range(2,n+1):
         if i <= n:
             res[i] = res[i-1] + res[i-2]
    return res[n]

# Given a set x ={x1,..., xN} of number of stairs the person can climb at once
# Using the same logic, we can see that:
# - the person can climb X1 stairs and will have (n-x1) remaining steps to climb
# - the person can climb X2 stairs and will have (n-x2) remaining steps to climb
# ...
# - the person can climb xN stairs and will have (n - xN) remaining steps to climb    
    
def NumberWays2(n,x):
    res =[0 for i in range(n+1)]
    res[0], res[1] =1,1
    for i in range(2,n+1):
        for j in x:
            if i <= n and j <=i :
                res[i] = res[i] + res[i-j] 
    return res[n]


## Test    
x={1,2}
n=100
print(NumberWays(n))
print(NumberWays2(n,x))       
