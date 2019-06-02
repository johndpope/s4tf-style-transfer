import TensorFlow
import Python
//import Path
import Foundation



public extension String {
    @discardableResult
    func shell(_ args: String...) -> String
    {
        let (task,pipe) = (Process(),Pipe())
        task.executableURL = URL(fileURLWithPath: self)
        (task.arguments,task.standardOutput) = (args,pipe)
        do    { try task.run() }
        catch { print("Unexpected error: \(error).") }

        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: String.Encoding.utf8) ?? ""
    }
}




// //// Add a gram matrix method to `Tensor`

// We use the gram matrix to extract the correlation structure between the filters of a given layer. This effectively allows us to decouple texture information from the global arrangement of the scene. We'll use this later when we calculate the perceptual loss.

// In[ ]:


public extension Tensor where Scalar: TensorFlowFloatingPoint {

    // Define the gram product.
    @differentiable
    func gramMatrix() -> Tensor<Scalar> {
        
        // Reshape to flatten the image dims to one dimension. Drop the batch dimension as well.
        let flatActivation = reshaped(to: TensorShape(shape[0], 
                                      shape[1] * shape[2], 
                                      shape[3])).squeezingShape(at: 0)
        
        // Take the matrix product of the transpose by the original. 
        return (flatActivation.transposed() • flatActivation) / Scalar(shape[1] * shape[2]) 
    }
}


// //// Define a layer for the input image

// **Disclaimer:** This is probably not the best way to handle this, but after trying a few alternatives, I landed on this. The idea is that we want to freeze the parameters of the Conv Net and only update the input image during back prop. At the time of writing this, I couldn't find a straightforward way to **A)** freeze a layer while still computing gradients and **B)** have the input image be a tunable parameter. I have some ideas on how I might be able to better handle this, but this works well enough for now.

// In[ ]:


////(/, This, layer, allows, us, to, update, the, input, image, during, back, propagation.)
////(/, Probably, not, ideal., See, above, fore, more, info.)
struct ImageTensorLayer: Layer {
    
    // This is the actual image that we'll be passing through the network.
    var imageTensor: Tensor<Float>
    
    init(imageTensor: Tensor<Float>) {
        self.imageTensor = imageTensor
    }
    
    //(/, Note, that, this, call, ignores, the, input, parameter, and, just, passes, imageTensor, through.)
    @differentiable
    func call(_ input: Tensor<Float>) -> Tensor<Float> {
        return self.imageTensor
    }
}


// //// Define a struct to store the output activations

// Swift for TensorFlow also doesn't currently support differentiable control flow which leads to more repetitious code. It also doesn't support pulling activations out of a net after a forward pass. There are definitely work arounds for both of these issues but they involve writing writing some less than straightforward workarounds. At some point this functionality will probably be baked into the Swift for TensorFlow deep learning framework.

// In[ ]:


////(/, This, is, used, to, store, the, output, activations., These, are, used, to, compute, the, perceptual, loss.)
////(/, As, the, property, names, suggest,, we'll, be, using, layers, 1a,, 2a,, 3a,, 4a, and, 5a, for, our, style, loss.)
////(/, and, layers, 4b, and, 5b, for, our, content, loss.)
struct OutputActivations: Differentiable {
    var activation1a: Tensor<Float>
    var style1a:      Tensor<Float>
    var activation2a: Tensor<Float>
    var style2a:      Tensor<Float>
    var activation3a: Tensor<Float>
    var style3a:      Tensor<Float>
    var activation4a: Tensor<Float>
    var style4a:      Tensor<Float>
    var activation4b: Tensor<Float>
    var activation5a: Tensor<Float>
    var style5a:      Tensor<Float>
    var activation5b: Tensor<Float>

    @differentiable
    init(activation1a: Tensor<Float>,
         activation2a: Tensor<Float>,
         activation3a: Tensor<Float>,
         activation4a: Tensor<Float>,
         activation4b: Tensor<Float>,
         activation5a: Tensor<Float>,
         activation5b: Tensor<Float>) {
        
        self.activation1a = activation1a
        self.style1a = activation1a.gramMatrix()
        
        self.activation2a = activation2a
        self.style2a = activation2a.gramMatrix()
        
        self.activation3a = activation3a
        self.style3a = activation3a.gramMatrix()
        
        self.activation4a = activation4a
        self.style4a = activation4a.gramMatrix()
        
        self.activation4b = activation4b
        
        self.activation5a = activation5a
        self.style5a = activation5a.gramMatrix()
        
        self.activation5b = activation5b
    }
}


// //// Define a concrete pooling type

// We'll use this to easily swap which type of pooling we're using. Again, once differentiable control flow lands, this sort of thing shouldn't be necessary. I tried getting this to work nicely with generics but to no avail. I looked to the way activation functions were implemented to achieve this.

// In[ ]:


enum PoolingType{
    case max
    case avg
}

struct PoolingLayer<Scalar: TensorFlowFloatingPoint>: Layer {    
    
    public typealias PoolingOperation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>
    
    @noDerivative let poolingOperation: PoolingOperation

    init(poolingType: PoolingType, 
         poolSize: (Int, Int) = (2, 2), 
         strides: (Int, Int) = (2, 2), 
         padding: Padding = .valid) {
        
        switch poolingType {
            case .max:
                let maxPool = MaxPool2D<Scalar>(poolSize: poolSize, strides: strides, padding: padding)
                poolingOperation = { (input: Tensor<Scalar>) -> Tensor<Scalar> in 
                                        return maxPool(input)
                                   }
            case .avg:
                let avgPool = AvgPool2D<Scalar>(poolSize: poolSize, strides: strides, padding: padding)
                poolingOperation = { (input: Tensor<Scalar>) -> Tensor<Scalar> in 
                                        return avgPool(input)
                                   }
        }
    }
    
    
    @differentiable
    func call(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
        return poolingOperation(input)
    }
}


// //// Implement the VGG19 architecture without the classification head

// This code should look fairly straightforward. Again, I could probably reduce some of the repition but this is just a first pass on this to get something up and running. I may do an update where I go in and refactor some of this (especially once differentiable control flow is supported).

// In[ ]:


struct VGG19: Layer {
    var conv1a: Conv2D<Float>
    var conv1b: Conv2D<Float>
    
    var conv2a: Conv2D<Float>
    var conv2b: Conv2D<Float>
    
    var conv3a: Conv2D<Float>
    var conv3b: Conv2D<Float>
    var conv3c: Conv2D<Float>
    var conv3d: Conv2D<Float>
    
    var conv4a: Conv2D<Float>
    var conv4b: Conv2D<Float>
    var conv4c: Conv2D<Float>
    var conv4d: Conv2D<Float>
    
    var conv5a: Conv2D<Float>
    var conv5b: Conv2D<Float>
    var conv5c: Conv2D<Float>
    var conv5d: Conv2D<Float>
    
    // Default to max pooling
    var poolingLayer = PoolingLayer<Float>(poolingType: .max)
    
    init() {
        // CheckpointReader defines an extension that reads the weights and biases for a given conv layer.
        // I just threw CheckpointReader together for this notebook so it's not super reusable, but proper
//        //(/, model, saving/loading/checkpointing, is, on, the, roadmap, so, it'll, be, easier, to, swap, to, that, once, it's)
//        //(/, available.)
        
        //(/, Layer, 1)
        self.conv1a = Conv2D(named: "block1_conv1")
        self.conv1b = Conv2D(named: "block1_conv2")
        
        //(/, Layer, 2)
        self.conv2a = Conv2D(named: "block2_conv1")
        self.conv2b = Conv2D(named: "block2_conv2")
        
        //(/, Layer, 3)
        self.conv3a = Conv2D(named: "block3_conv1")
        self.conv3b = Conv2D(named: "block3_conv2")
        self.conv3c = Conv2D(named: "block3_conv3")
        self.conv3d = Conv2D(named: "block3_conv4")
        
        //(/, Layer, 4)
        self.conv4a = Conv2D(named: "block4_conv1")
        self.conv4b = Conv2D(named: "block4_conv2")
        self.conv4c = Conv2D(named: "block4_conv3")
        self.conv4d = Conv2D(named: "block4_conv4")
        
        //(/, Layer, 5)
        self.conv5a = Conv2D(named: "block5_conv1")
        self.conv5b = Conv2D(named: "block5_conv2")
        self.conv5c = Conv2D(named: "block5_conv3")
        self.conv5d = Conv2D(named: "block5_conv4")
    }
    
    @differentiable
    func call(_ input: Tensor<Float>) -> OutputActivations {
        var tmp = input
        
        // Layer 1
        tmp = conv1a(tmp)
        let act1a = tmp
        tmp = conv1b(tmp)
        tmp = poolingLayer(tmp)
        
        // Layer 2 
        tmp = conv2a(tmp)
        let act2a = tmp
        tmp = conv2b(tmp)
        tmp = poolingLayer(tmp)
        
        // Layer 3
        tmp = conv3a(tmp)
        let act3a = tmp
        tmp = conv3b(tmp)
        tmp = conv3c(tmp)
        tmp = conv3d(tmp)
        tmp = poolingLayer(tmp)
        
        // Layer 4
        tmp = conv4a(tmp)
        let act4a = tmp
        tmp = conv4b(tmp)
        let act4b = tmp
        tmp = conv4c(tmp)
        tmp = conv4d(tmp)
        tmp = poolingLayer( tmp)
        
        // Layer 5
        tmp = conv5a(tmp)
        let act5a = tmp
        tmp = conv5b(tmp)
        let act5b = tmp
        tmp = conv5c(tmp)
        tmp = conv5d(tmp)
        tmp = poolingLayer(tmp)
        
        return OutputActivations(activation1a: act1a, 
                                 activation2a: act2a, 
                                 activation3a: act3a, 
                                 activation4a: act4a, 
                                 activation4b: act4b, 
                                 activation5a: act5a,
                                 activation5b: act5b)
    }
}


// //// Tie the two layers together

// I chose to compose the two layers into one here. I had written this prior to S4TF 0.3 which I believe introduced a better way to sequence layers together. I'll probably revisit this also.

// In[ ]:


struct StyleTransferModel: Layer {
    var inputLayer: ImageTensorLayer
    var model: VGG19
    
    init(inputLayer: ImageTensorLayer, model: VGG19) {
        self.inputLayer = inputLayer
        self.model = model
    }
    
    @differentiable
    func call(_ input: Tensor<Float>) -> OutputActivations {
        let image = self.inputLayer(input)
        return model(image)
    }
}


// //// Define an optimizer

// This is probably the hackiest part of this. I'm sure there's a way to use the existing Adam optimizer and freeze all the parameters in the network, updating only the image, however I couldn't figure it out and this worked. I'll definitely revisit this on my next pass.
//
// This code is identical to the Adam definition in the S4TF Deep Learning library except for the fact that it breaks out of the for loop after updating the first differentiable variable. The first differentiable variable is the image.

// In[ ]:


func zeroTensor(tensor: Tensor<Float>) -> Tensor<Float> {
    return Tensor<Float>(zeros: tensor.shape)
}

//(//, Adam, optimizer.)
//(//)
//(//, Reference:, ["Adam, -, A, Method, for, Stochastic, Optimization"]()
/// https://arxiv.org/abs/1412.6980v8)
public class ImageAdam<Model: Layer>: Optimizer
    where Model.AllDifferentiableVariables == Model.CotangentVector {
    /// The learning rate.
    public var learningRate: Float
    /// A coefficient used to calculate the first and second moments of
    /// gradients.
    public var beta1: Float
    /// A coefficient used to calculate the first and second moments of
    /// gradients.
    public var beta2: Float
    /// A small scalar added to the denominator to improve numerical stability.
    public var epsilon: Float
    /// The weight decay.
    public var decay: Float
    /// The current step.
    public var step: Int = 0
    /// The first moments of the weights.
    public var firstMoments: Model.AllDifferentiableVariables
    /// The second moments of the weights.
    public var secondMoments: Model.AllDifferentiableVariables

    public init(
        for model: __shared Model,
        learningRate: Float = 1e-3,
        beta1: Float = 0.9,
        beta2: Float = 0.999,
        epsilon: Float = 1e-8,
        decay: Float = 0
    ) {
        precondition(learningRate >= 0, "Learning rate must be non-negative")
        precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
        precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")
        precondition(decay >= 0, "Weight decay must be non-negative")

        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay

        // Initialize first & second moments to be zeros of the same shape.
        // We can't use `Model.AllDifferentiableVariables.zero` due to the
        //(/, interaction, between, Key, Paths, and, Differentiable, Arrays.)
        firstMoments = model.allDifferentiableVariables
        secondMoments = model.allDifferentiableVariables
        for kp in firstMoments.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            firstMoments[keyPath: kp] = zeroTensor(tensor: firstMoments[keyPath: kp])
            secondMoments[keyPath: kp] = zeroTensor(tensor: secondMoments[keyPath: kp])
        }
    }


    public func update(_ model: inout Model.AllDifferentiableVariables,
                       along direction: Model.AllDifferentiableVariables) {
        step += 1
        let learningRate = self.learningRate * 1 / (1 + decay * Float(step))
        let stepSize = learningRate * (sqrt(1 - pow(beta2, Float(step))) /
            (1 - pow(beta1, Float(step))))
        
        // Update Float Tensor variables.
        for kp in model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            firstMoments[keyPath: kp] =
                firstMoments[keyPath: kp] * beta1 + (1 - beta1) * direction[keyPath: kp]
            secondMoments[keyPath: kp] =
                secondMoments[keyPath: kp] * beta2 + (1 - beta2) *
                direction[keyPath: kp] * direction[keyPath: kp]
            model[keyPath: kp] -=
                stepSize * firstMoments[keyPath: kp] / (sqrt(secondMoments[keyPath: kp]) + epsilon)
            break
        }
    }
}


// //// Total variation loss

// Total variation loss is a regularization technique that is commonly used to denoise images. Unfortunately this currently causes the GPU memory to grow unbounded and triggers an OOM error. There's probably a clever way to avoid this, but the results look pretty good without it. I've kept it here for reference--perhaps someone can spot what I'm doing wrong and let me know ;)

// In[ ]:


@differentiable
public func totalVariationLoss(imageTensor: Tensor<Float>) -> Tensor<Float> { 
    
    let rank = imageTensor.rank
    let shape = imageTensor.shape
        
    let diff1LowerUp = [0, 1, 0, 0]
    let diff1UpperUp = [1, shape[1], shape[2], shape[3]]
    
    let diff1LowerDown = [0, 0, 0, 0]
    let diff1UpperDown = [1, shape[1] - 1, shape[2], shape[3]]
    
    let diff2LowerUp = [0, 0, 1, 0]
    let diff2UpperUp = [1, shape[1], shape[2], shape[3]]
    
    let diff2LowerDown = [0, 0, 0, 0]
    let diff2UpperDown = [1, shape[1], shape[2] - 1, shape[3]]
    
    let pixelDiff1 = imageTensor.slice(lowerBounds: diff1LowerUp, upperBounds: diff1UpperUp)
                     - imageTensor.slice(lowerBounds: diff1LowerDown, upperBounds: diff1UpperDown)
    
    let pixelDiff2 = imageTensor.slice(lowerBounds: diff2LowerUp, upperBounds: diff2UpperUp)
                     - imageTensor.slice(lowerBounds: diff2LowerDown, upperBounds: diff2UpperDown)
    
    return (pixelDiff1 * pixelDiff1).mean() + (pixelDiff2 * pixelDiff2).mean()
}


// //// Perceptual Loss

// This is the loss function we'll use. We compute the **mean squared error** (MSE) between the target content activations and the styled image. We then do the same for the style activation layers but instead of computing the MSE of the raw activations, we compute the MSE of the **gram matrix** of the activations. Each style and content layer loss is then scaled by its own weight. Lastly, every thing is summed up and returned as our final loss.

// In[ ]:


@differentiable(wrt: styledImageActivations)
func perceptualLoss(contentTargetActivations: OutputActivations, 
                    styleTargetActivations: OutputActivations,
                    styledImageActivations: OutputActivations,
                    contentWeights: [Float],
                    styleWeights: [Float]) -> Tensor<Float> {
    
    // Content Loss
    var loss = meanSquaredError(predicted: styledImageActivations.activation4b, 
                                expected: contentTargetActivations.activation4b) * contentWeights[0]
    
    loss = loss + meanSquaredError(predicted: styledImageActivations.activation5b, 
                                  expected: contentTargetActivations.activation5b) * contentWeights[1]
    
    // Style Loss
    loss = loss + meanSquaredError(predicted: styledImageActivations.style1a, 
                                   expected: styleTargetActivations.style1a) * styleWeights[0]
    
    loss = loss + meanSquaredError(predicted: styledImageActivations.style2a, 
                                   expected: styleTargetActivations.style2a) * styleWeights[1]
    
    loss = loss + meanSquaredError(predicted: styledImageActivations.style3a, 
                                   expected: styleTargetActivations.style3a) * styleWeights[2]
    
    loss = loss + meanSquaredError(predicted: styledImageActivations.style4a, 
                                   expected: styleTargetActivations.style4a) * styleWeights[3]
    
    loss = loss + meanSquaredError(predicted: styledImageActivations.style5a, 
                                   expected: styleTargetActivations.style5a) * styleWeights[4]
    
    return loss
}


// //// A clamping utility

// This utility is used to keep the color values in the range that VGG19 expects to see, i.e. +/- the imagenet mean pixel values (in BGR). Without this, regions of the image would over excite the network causing clipped regions and aberrant noise. **Note:** It might be an interesting experiment to clamp these values with a smooth function instead of min/max. It might reduce noise a bit.

// In[ ]:


func clamp(image: Tensor<Float>, to mean: Tensor<Float>) -> Tensor<Float> {
    let maxTensor = mean.broadcast(like: image)
    let minTensor = -maxTensor
    
    let clampedImage = max(min(image, maxTensor), minTensor)
    
    return clampedImage
}

