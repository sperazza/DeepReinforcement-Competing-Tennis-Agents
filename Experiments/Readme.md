This directory contains some scratch experiement results:

- Tennis.ipynb  -  This is a standard MLP NN with no prev_state-state feature modification
- Tennis_3LayerMLP - This is a 3 Layer MLP with prev_state-state feature modification
    - converges successfully, but runs slower
- Tennis_onehot  -  This is a standard MLP, with the following one hot vector concatenated to feature input:
```
        self.one_arr=np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1])
        self.zero_arr = np.array([1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.one_zero=np.array([self.one_arr,self.zero_arr])
 ```
    - this converged ok,but was also slower than optimal solution
