import TensorFlow
import Python
//import Path
import Foundation


// WARNING - if you get a crash - check which python - it maybe under miniconda3 or miniconda3!
// N.B. - this branch is using miniconda3 not virtualenv see
// https://github.com/johndpope/SwiftReinforce readme
let  path = "/Users/\(NSUserName())/miniconda3/envs/gymai2/lib/python2.7/site-packages/"

// IMPORTANT - if you're using a different environment name
// Schema > Run > Pre-actions - source activate gymai2
let environmentName = "swift-tensorflow"


let sys = Python.import("sys")
print("INFO: macOS sysPath:", sys.path)
let  path2 = "/Users/johndpope/miniconda3/envs/\(environmentName)/lib/python3.6/site-packages/"
sys.path.insert(0,path2)

let np = Python.import("numpy")
let display = Python.import("IPython.display")


// //// Image processing utilities

// These functions are used to load up the images into Float tensors and pre/post process them. This pretrained VGG-19 was trained on images in BGR channel order that were normalized by the imagenet mean. Note that they did not divide by the standard deviation so values be in the range of **`[-mean, 255 - mean]`**.

// In[ ]:


func loadImage(fileName: StringTensor) -> Tensor<UInt8> {
    let imageBytes: StringTensor = Raw.readFile(filename: fileName)
    return Raw.decodeJpeg(contents: imageBytes, dctMethod: "")
}


// In[ ]:


func centerCrop<Scalar: TensorFlowNumeric>(image: Tensor<Scalar>, size: Int32) -> Tensor<Float> {
    // Use Raw.cropAndResize() to crop a square of (size x size) from the center of the image.
    let rank = image.rank
    precondition(rank == 4, "Image must be of rank 4 but image is of rank \(rank).")
    let (height, width) = (Float(image.shape[rank - 3]), Float(image.shape[rank - 2]))
    
    precondition(width * height > Float(size * size), 
                 "Image of size: (\(height), \(width)) already smaller than \(size).")
    
    let verticalSpace: Float = (height - Float(size)) / Float(2.0)
    let horizontalSpace: Float = (width - Float(size)) / Float(2.0)
    
    let y2 = (height - verticalSpace) / Float(height - 1.0)
    let x2 = (width - horizontalSpace) / Float(width - 1.0)
    let y1 = verticalSpace / Float(height - 1.0)
    let x1 = horizontalSpace / Float(width - 1.0)
        
    let boxes = Tensor<Float>([y1, x1, y2, x2]).expandingShape(at: 0)
    let cropSize = Tensor<Int32>([size, size])
    
    return Raw.cropAndResize(image: image, 
                             boxes: boxes, 
                             boxInd: [0], 
                             cropSize: cropSize)    
}


// In[ ]:


enum ByteOrdering {
    case bgr
    case rgb
}


// In[ ]:


func preprocess(image: Tensor<UInt8>,
                size: Int32,
                inByteOrdering: ByteOrdering,
                outByteOrdering: ByteOrdering,
                meanToSubtract: Tensor<Float>) -> Tensor<Float> {
    
    let rank = image.rank
    let (height, width) = (Float(image.shape[rank - 3]), Float(image.shape[rank - 2]))
    
    var resizedImage = Tensor<Float>(image.expandingShape(at: 0))
    
    if width * height > Float(size * size) {
        resizedImage = centerCrop(image: resizedImage, size: size)
    } else {
        let sizeTensor = Tensor<Int32>([size, size])
        var resizedImage = Tensor<Float>(Raw.resizeNearestNeighbor(images: resizedImage, 
                                                                   size: sizeTensor))        
    }
        
    if inByteOrdering != outByteOrdering {
        resizedImage = Raw.reverse(resizedImage, dims: Tensor<Bool>([false, false, false, true]))
    }
    
    return resizedImage - meanToSubtract.expandingShape(at: 0)
}

func postprocess(image: Tensor<Float>,
                 inByteOrdering: ByteOrdering,
                 outByteOrdering: ByteOrdering, 
                 meanToAdd: Tensor<Float>) -> Tensor<UInt8> {

    var processedImage = image + meanToAdd
    
    if inByteOrdering != outByteOrdering {
        processedImage = Raw.reverse(processedImage, dims: Tensor<Bool>([false, false, false, true]))
    }

    return Tensor<UInt8>(processedImage).squeezingShape(at: 0)
}


// //// Set up matplotlib for inline display

// In[ ]:


//(/, Setup.)
//get_ipython().run_line_magic('include', '"EnableIPythonDisplay.swift"')
let plt = Python.import("matplotlib.pyplot")
//IPythonDisplay.shell.enable_matplotlib("inline")


// In[ ]:


let imageNetMean = Tensor<Float>([116.779, 103.939, 123.68])


// //// A utility to display an image tensor

// In[ ]:


func showImageTensor(tensor: Tensor<Float>,
                     byteOrdering: ByteOrdering) {
    plt.figure(figsize: [10, 10])
    plt.axis("off")
    let pixelTensor = postprocess(image: tensor,
                                  inByteOrdering: byteOrdering, 
                                  outByteOrdering: .rgb, 
                                  meanToAdd: imageNetMean)
    plt.imshow(pixelTensor.makeNumpyArray())
    plt.show()
}


// //// Define a struct to hold the training results

// This is where the results of training get stored so we can see how the training progressed over time. It also has a function that will plot the output images in a grid. This is nifty when trying out and comparing different hyper parameters.

// In[ ]:


struct StyleTransferResult: CustomStringConvertible {
    var outputImages: [Tensor<Float>] = []
    var losses: [Float] = []
    
    let styleWeights: [Float]
    let contentWeights: [Float]
    let lr: Float
    let iterations: Int
    let saveEvery: Int
    
    var description: String {
        let description = """
               contentWeights: \(contentWeights)\n 
               styleWeights: \(styleWeights)\n
               lr: \(lr)\n
               iterations: \(iterations)\n
               saveEvery: \(saveEvery)\n
               losses: \(losses)\n
               """
        return description
    }
    
    init(styleWeights: [Float], 
         contentWeights: [Float], 
         lr: Float, 
         iterations: Int,
         saveEvery: Int) {
        self.styleWeights = styleWeights
        self.contentWeights = contentWeights
        self.lr = lr
        self.iterations = iterations
        self.saveEvery = saveEvery
    }
    
    public func showImages() {
        let cols = 2
        let rows = Int(ceil(Float(outputImages.count) / Float(cols)))
        
        var subplotsTuple = plt.subplots(ncols: cols, 
                                      nrows: rows,
                                      figsize: [14, 7*rows])
        
        let figure = subplotsTuple[0]
        let ax = subplotsTuple[1]
        
        for i in 0..<(rows * cols) {
            let r = i / cols
            let c = i % cols
            
            ax[r][c].axis("off")
            
            if i >= outputImages.count { break }

            let pixelTensor = postprocess(image: outputImages[i],
                                          inByteOrdering: .bgr,
                                          outByteOrdering: .rgb,
                                          meanToAdd: imageNetMean)

            let x = pixelTensor.makeNumpyArray()

            ax[r][c].imshow(x)
            ax[r][c].set_title("Iteration: \(i * 50)")
        }
        
        plt.show()
    }
}


// //// Peek at the style and content images

// We'll use the images from the graphic at the beginning of the post. **Note**: If you run into memory issues, you can drop the size down to 256.

// In[ ]:


let contentImageBytes = loadImage(fileName:StringTensor("./painted_ladies.jpg"))
let contentImageTensor = preprocess(image: contentImageBytes, 
                                    size: 512, 
                                    inByteOrdering: .rgb, 
                                    outByteOrdering: .bgr,
                                    meanToSubtract: imageNetMean)

let styleImageBytes = loadImage(fileName:StringTensor("./vangogh_starry_night.jpg"))
let styleImageTensor = preprocess(image: styleImageBytes, 
                                  size: 512, 
                                  inByteOrdering: .rgb, 
                                  outByteOrdering: .bgr, 
                                  meanToSubtract: imageNetMean)


// In[ ]:


print(contentImageTensor.shape)
print(styleImageTensor.shape)


// In[ ]:


showImageTensor(tensor: contentImageTensor, byteOrdering: .bgr)
showImageTensor(tensor: styleImageTensor, byteOrdering: .bgr)


// //// Define the training method

// Right now I've only really needed to change the style weights, content weight, iteration count and learning rate, so that's what's exposed.

// In[ ]:


func train(model: inout StyleTransferModel,
           contentTarget: OutputActivations,
           styleTarget: OutputActivations,
           lr: Float, 
           iterations: Int = 450, 
           contentWeights: [Float],
           styleWeights: [Float],
           saveEvery: Int) -> StyleTransferResult {
    
    // Set up the optimizer
    let targetOptimizer = ImageAdam(for: model, learningRate: lr, beta1: 0.9, beta2: 0.999, epsilon: 1e-8)
    
    // Set up the result where we'll store training progress.
    var result = StyleTransferResult(styleWeights: styleWeights, 
                                     contentWeights: contentWeights, 
                                     lr: lr, 
                                     iterations: iterations,
                                     saveEvery: saveEvery)
    
    //(/, Loop, for, the, specified, iterations.)
    for i in 0..<iterations {
        
        // Keep track of the loss.
        var lastLoss: Float = 0.0
        
        // Compute the cotangent vector of the loss with respect to the styled image.
        let 𝛁model = model.gradient { model -> Tensor<Float> in
            
            //(/, Run, the, forward, pass, for, the, model.)
            let styledImageActivations = model(model.inputLayer.imageTensor)
            //(/, Compute, the, perceptual, loss)
            var loss = perceptualLoss(contentTargetActivations: contentTarget, 
                                      styleTargetActivations: styleTarget, 
                                      styledImageActivations: styledImageActivations,
                                      contentWeights: contentWeights,
                                      styleWeights: styleWeights)
                                      
            //(/, Store, the, loss, so, we, can, track, it, outside, of, this, closure.)
            lastLoss = loss.scalarized()
                                      
            return loss
        }
        
        //(/, Update, the, model, along, the, cotangent, vector)
        targetOptimizer.update(&model.allDifferentiableVariables, along: 𝛁model)
        
        //(/, Clamp, the, styled, image, to, be, in, the, range, of, [-mean,, mean].)
        model.inputLayer.imageTensor = clamp(image: model.inputLayer.imageTensor, to: imageNetMean)
        
        //(/, Save, the, output, image, if, necessary.)
        if i % saveEvery == 0  {
            print("\t[Iteration \(i) - Perceptual Loss: \(lastLoss)]")
            result.outputImages.append(model.inputLayer.imageTensor)
            result.losses.append(lastLoss)
        } else if i == iterations - 1 {
            print("\t[Iteration \(iterations) - Perceptual Loss: \(lastLoss)]")
            result.outputImages.append(model.inputLayer.imageTensor)
            result.losses.append(lastLoss)
        }
    }
    
    return result
}


// //// Tie things up into a function we can experiment with

// In[ ]:


func styleTransfer(styleImagePath: String,
                   contentImagePath: String,
                   styledImage: Tensor<Float>? = nil,
                   imageSize: Int32 = 256,
                   poolingType: PoolingType = .max,
                   learningRate: Float = 4.0,
                   styleWeights: [Float],
                   contentWeights: [Float],
                   iterations: Int = 500,
                   saveEvery: Int = 50) -> StyleTransferResult {

    //(/, Load, up, the, specified, style, and, content, images.)
    let contentImageBytes = loadImage(fileName:StringTensor(contentImagePath))
    let contentImageTensor = preprocess(image: contentImageBytes, 
                                        size: imageSize, 
                                        inByteOrdering: .rgb, 
                                        outByteOrdering: .bgr,
                                        meanToSubtract: imageNetMean)
    
    let styleImageBytes = loadImage(fileName:StringTensor(styleImagePath))
    let styleImageTensor = preprocess(image: styleImageBytes, 
                                      size: imageSize, 
                                      inByteOrdering: .rgb, 
                                      outByteOrdering: .bgr, 
                                      meanToSubtract: imageNetMean)
    
    //(/, Set, up, the, base, VGG19, model)
    var baseModel = VGG19()
    
    //(/, Set, the, pooling, type, based, on, user, preference.)
    baseModel.poolingLayer = PoolingLayer(poolingType: poolingType)
    
    //(/, Compute, the, target, activations, for, the, content, and, style.)
    let contentTarget = baseModel.inferring(from: contentImageTensor)
    let styleTarget = baseModel.inferring(from: styleImageTensor)
    
    //(/, If, an, initial, styledImage, was, passed, in,, use, it;, otherwise, use, the, content, image.)
    let styledImageTensor = styledImage ?? contentImageTensor
    
    //(/, Set, up, the, net, for, optimizing, the, styled, image.)
    let styledInputLayer = ImageTensorLayer(imageTensor: styledImageTensor)
    var styleTransferModel = StyleTransferModel(inputLayer: styledInputLayer, model: baseModel)
    
    //(/, Optimize, the, image!)
    return train(model: &styleTransferModel,
                 contentTarget: contentTarget,
                 styleTarget: styleTarget,
                 lr: learningRate, 
                 iterations: iterations,
                 contentWeights: contentWeights,
                 styleWeights: styleWeights,
                 saveEvery: saveEvery)
}


// //// Let's finally perform style transfer to make an image

// In[ ]:


//(/, Keep, an, array, of, results, for, comparison)
var results: [StyleTransferResult] = []


// We'll just use most of the default params for the style transfer function.

// In[ ]:


var styleWeights = [Float(1e3 / Float(pow(64.0 , 2.0))),
                    Float(1e3 / Float(pow(128.0, 2.0))),
                    Float(1e3 / Float(pow(256.0, 2.0))),
                    Float(1e3 / Float(pow(512.0, 2.0))),
                    Float(1e3 / Float(pow(512.0, 2.0)))]

var contentWeights:[Float] = [1.0, 1.0]

var result = styleTransfer(styleImagePath: "./vangogh_starry_night.jpg",
                           contentImagePath: "./painted_ladies.jpg", 
                           imageSize: 512,
                           styleWeights: styleWeights,
                           contentWeights: contentWeights)

results.append(result)

//(/, Look, at, the, last, image.)
showImageTensor(tensor: results.last!.outputImages.last!, byteOrdering: .bgr)


// Pretty cool, huh? The original paper uses [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) which is a pretty memory intensive optimization algorithm (at least compared to standard NN optimzation algorithms like Adam). L-BFGS tends to produce better results, but I think this looks pretty good. Of couse we'd need to try it on more than one style/content pair to make a fair assessment.

// //// Let's look at how the image progressed over time

// Notice how it actually converges to something pretty reasonable after about 100 iterations

// In[ ]:


result.showImages()


// //// Now let's try initializing with the style image instead

// We'll use higher content weights this time around. As one might imagine starting with the style image will bias towards the global structure of the style image. We could alternatively tune the style weights down a bit, but let's see where this gets us.

// In[ ]:


contentWeights = [25.0, 35.0]

result = styleTransfer(styleImagePath: "./vangogh_starry_night.jpg",
                           contentImagePath: "./painted_ladies.jpg",
                           styledImage: styleImageTensor,
                           imageSize: 512,                           
                           styleWeights: styleWeights,
                           contentWeights: contentWeights)

results.append(result)

//(/, Look, at, the, last, image.)
showImageTensor(tensor: results.last!.outputImages.last!, byteOrdering: .bgr)


// So as you can tell, the global structure of the style image is still very present. This is definitely a more abstract result!

// //// Let's try to tune the style weights

// In[ ]:


//(/, Let's, nudge, down, the, higher, layers.)
styleWeights = [Float(1e3 / Float(pow(64.0 , 2.0))),
                    Float(1e3 / Float(pow(128.0, 2.0))),
                    Float(1e3 / Float(pow(256.0, 2.0))),
                    Float(1e3 / Float(pow(1024.0, 2.0))), // was 1e3 / 512^2
                    Float(1e3 / Float(pow(1024.0, 2.0)))] // was 1e3 / 512^2

contentWeights = [25.0, 35.0]

result = styleTransfer(styleImagePath: "./vangogh_starry_night.jpg",
                           contentImagePath: "./painted_ladies.jpg",
                           styledImage: styleImageTensor,
                           imageSize: 512,
                           styleWeights: styleWeights,
                           contentWeights: contentWeights)

results.append(result)

//(/, Look, at, the, last, image.)
showImageTensor(tensor: results.last!.outputImages.last!, byteOrdering: .bgr)


// This definitely produces a clearer result. It's worth showing the content image again...

// In[ ]:


showImageTensor(tensor: contentImageTensor, byteOrdering: .bgr)


// //// Let's try using average instead of max pooling

// In[ ]:


//(/, Let's, nudge, down, the, higher, layers.)
styleWeights = [Float(1e3 / Float(pow(64.0 , 2.0))),
                    Float(1e3 / Float(pow(128.0, 2.0))),
                    Float(1e3 / Float(pow(256.0, 2.0))),
                    Float(1e3 / Float(pow(512.0, 2.0))),
                    Float(1e3 / Float(pow(512.0, 2.0)))] 

contentWeights = [3.0, 3.0]

result = styleTransfer(styleImagePath: "./vangogh_starry_night.jpg",
                           contentImagePath: "./painted_ladies.jpg",
                           styledImage: styleImageTensor,
                           imageSize: 512,
                           poolingType: .avg,
                           styleWeights: styleWeights,
                           contentWeights: contentWeights)

results.append(result)

//(/, Look, at, the, last, image.)
showImageTensor(tensor: results.last!.outputImages.last!, byteOrdering: .bgr)


// I think average pooling might be less noisy. This looks pretty good, though.

// //// To close out, let's test on some new images

// Here's our style image.

// In[ ]:


let styleImage = loadImage(fileName: StringTensor("./vangogh_rhone.jpg"))
let styleTensor = preprocess(image: styleImage, 
                              size: 512, 
                              inByteOrdering: .rgb, 
                              outByteOrdering: .bgr, 
                              meanToSubtract: imageNetMean)
showImageTensor(tensor: styleTensor, byteOrdering: .bgr)


// Starry Night Over the Rhône by Vincent van Gogh

// We'll apply it to this Image of the Oakland Bay Bridge.

// In[ ]:


let contentImage = loadImage(fileName: StringTensor("./baybridge.jpg"))
let contentTensor = preprocess(image: contentImage, 
                                size: 512, 
                                inByteOrdering: .rgb, 
                                outByteOrdering: .bgr, 
                                meanToSubtract: imageNetMean)

showImageTensor(tensor: contentTensor, byteOrdering: .bgr)


// Dllu [CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0)]

// In[ ]:


styleWeights = [Float(1e3 / Float(pow(64.0 , 2.0))),
                    Float(1e3 / Float(pow(64.0, 2.0))),
                    Float(1e3 / Float(pow(256.0, 2.0))),
                    Float(1e3 / Float(pow(512.0, 2.0))),
                    Float(1e3 / Float(pow(1024.0, 2.0)))]

contentWeights = [55.0, 55.0]

//(/, We'll, init, with, the, style, image, again.)
result = styleTransfer(styleImagePath: "./vangogh_rhone.jpg",
                           contentImagePath: "./baybridge.jpg", 
                           styledImage: styleTensor,
                           imageSize: 512,
                           learningRate: 8,
                           styleWeights: styleWeights,
                           contentWeights: contentWeights)

results.append(result)

//(/, Look, at, the, last, image.)
showImageTensor(tensor: results.last!.outputImages.last!, byteOrdering: .bgr)


// In[ ]:




