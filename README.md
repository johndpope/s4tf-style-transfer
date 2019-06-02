# s4tf-style-transfer
Neural Style Transfer using Swift for TensorFlow
https://github.com/regrettable-username/style-transfer


Download / extract vgg19 folder https://github.com/regrettable-username/style-transfer/releases/download/v0.1/vgg19.tar.gz into project folder    

N.B. conda environment must be set to 3.7 to align with S4TF
```shell
conda create -n swift-tensorflow python==3.7
conda activate swift-tensorflow
conda install jupyter numpy matplotlib
```

change the main.swift path to point to where it will find conda packages eg.   

```swift
let environmentName = "swift-tensorflow"
let  path = "/Users/YOURUSERNAMEHERE/miniconda3/envs/\(environmentName)/lib/python3.7/site-packages/"    
```



