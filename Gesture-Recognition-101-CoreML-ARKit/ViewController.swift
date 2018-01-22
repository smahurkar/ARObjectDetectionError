//
//  ViewController.swift
//  Gesture-Recognition-101-CoreML-ARKit
//
//  Created by Hanley Weng on 10/22/17.
//  Copyright © 2017 Emerging Interactions. All rights reserved.
//

import UIKit
import SceneKit
import ARKit
import Vision

class ViewController: UIViewController, ARSCNViewDelegate {

    @IBOutlet var sceneView: ARSCNView!
    @IBOutlet weak var debugTextView: UITextView!
    @IBOutlet weak var textOverlay: UITextField!
    
    let dispatchQueueML = DispatchQueue(label: "com.hw.dispatchqueueml") // A Serial Queue
    var visionRequests = [VNCoreMLRequest]()
    
    //struct to handle multi arrays
    struct Prediction {
        let labelIndex: Int
        let confidence: Float
        let boundingBox: CGRect
    }
    
    /// - Tag: MLModelSetup
    lazy var classificationRequest: VNCoreMLRequest = {
        do {
            //setup from turicreate
            let mlmodel = MyCustomObjectDetector()
            let userDefined: [String: String] = mlmodel.model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey]! as! [String : String]
            let labels = userDefined["classes"]!.components(separatedBy: ",")
            
            let nmsThreshold = Float(userDefined["non_maximum_suppression_threshold"]!) ?? 0.5
            
            /*
             Use the Swift class `MobileNet` Core ML generates from the model.
             To use a different Core ML classifier model, add it to the project
             and replace `MobileNet` with that model's generated Swift class.
             */
            let model = try VNCoreMLModel(for: mlmodel.model)
            
            let request = VNCoreMLRequest(model: model, completionHandler: { [weak self] request, error in
                self?.classificationCompleteHandler(for:request, error:error)
            })
            request.imageCropAndScaleOption = .scaleFill
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // --- ARKIT ---
        
        // Set the view's delegate
        sceneView.delegate = self
        
        // Show statistics such as fps and timing information
        sceneView.showsStatistics = true
        
        // Create a new scene
        let scene = SCNScene() // SCNScene(named: "art.scnassets/ship.scn")!
        
        // Set the scene to the view
        sceneView.scene = scene
        
        // --- ML & VISION ---
        
        //setup from turicreate
        let mlmodel = MyCustomObjectDetector()
        let userDefined: [String: String] = mlmodel.model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey]! as! [String : String]

        
//        // Set up Vision-CoreML Request
//        let classificationRequest = VNCoreMLRequest(model: selectedModel, completionHandler: classificationCompleteHandler)
//        classificationRequest.imageCropAndScaleOption = .scaleFill // Crop from centre of images and scale to appropriate size.
//        visionRequests = [classificationRequest]
        
        // Begin Loop to Update CoreML
        loopCoreMLUpdate()
    }
    

    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Create a session configuration
        let configuration = ARWorldTrackingConfiguration()

        // Run the view's session
        sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Pause the view's session
        sceneView.session.pause()
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Release any cached data, images, etc that aren't in use.
    }

    // MARK: - ARSCNViewDelegate
    
    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        DispatchQueue.main.async {
            // Do any desired updates to SceneKit here.
        }
    }
    
    // MARK: - MACHINE LEARNING
    
    func loopCoreMLUpdate() {
        // Continuously run CoreML whenever it's ready. (Preventing 'hiccups' in Frame Rate)
        dispatchQueueML.async {
            // 1. Run Update.
                self.updateCoreML()
            // 2. Loop this function.
                self.loopCoreMLUpdate()
        }
    }
    
    func updateCoreML() {
//        // Get Camera Image as RGB
//        let pixbuff : CVPixelBuffer? = (sceneView.session.currentFrame?.capturedImage)
//        if pixbuff == nil { return }
//        let ciImage = CIImage(cvPixelBuffer: pixbuff!)
        
        var image = sceneView.snapshot()
        let orientation = CGImagePropertyOrientation(image.imageOrientation)
        guard let ciImage = CIImage(image: image) else { fatalError("Unable to create \(CIImage.self) from \(image).") }
        
        
        
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
            do {
                try handler.perform([self.classificationRequest])
            } catch {
                /*
                 This handler catches general image processing errors. The `classificationRequest`'s
                 completion handler `processClassifications(_:error:)` catches errors specific
                 to processing that request.
                 */
                print("Failed to perform classification.\n\(error.localizedDescription)")
            }
        }
        
//        // Prepare CoreML/Vision Request
//        let imageRequestHandler = VNImageRequestHandler(ciImage: ciImage, options: [:])
//        // Run Vision Image Request
//        do {
//            try imageRequestHandler.perform([self.classificationRequest])
//        } catch {
//            print(error)
//        }
//
    }
    
    public func IoU(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        let union = a.union(b)
        return Float((intersection.width * intersection.height) / (union.width * union.height))
    }
    
    
    func classificationCompleteHandler(for request: VNRequest, error: Error?) {
        // Catch Errors
        if error != nil {
            print("Error: " + (error?.localizedDescription)!)
            return
        }
        
        
        
        // Render Classifications
        DispatchQueue.main.async {
            
            let results = request.results as! [VNCoreMLFeatureValueObservation]
            
            let mlmodel = MyCustomObjectDetector()
            
            let userDefined: [String: String] = mlmodel.model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey]! as! [String : String]
            let labels = userDefined["classes"]!.components(separatedBy: ",")
            
            let coordinates = results[0].featureValue.multiArrayValue!
            let confidence = results[1].featureValue.multiArrayValue!
            
            let confidenceThreshold = 0.1
            var unorderedPredictions = [Prediction]()
            let numBoundingBoxes = confidence.shape[0].intValue
            let numClasses = confidence.shape[1].intValue
            let confidencePointer = UnsafeMutablePointer<Double>(OpaquePointer(confidence.dataPointer))
            let coordinatesPointer = UnsafeMutablePointer<Double>(OpaquePointer(coordinates.dataPointer))
            for b in 0..<numBoundingBoxes {
                var maxConfidence = 0.0
                var maxIndex = 0
                for c in 0..<numClasses {
                    let conf = confidencePointer[b * numClasses + c]
                    if conf > maxConfidence {
                        maxConfidence = conf
                        maxIndex = c
                    }
                }
                if maxConfidence > confidenceThreshold {
                    let x = coordinatesPointer[b * 4]
                    let y = coordinatesPointer[b * 4 + 1]
                    let w = coordinatesPointer[b * 4 + 2]
                    let h = coordinatesPointer[b * 4 + 3]
                    
                    let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                      width: CGFloat(w), height: CGFloat(h))
                    
                    let prediction = Prediction(labelIndex: maxIndex,
                                                confidence: Float(maxConfidence),
                                                boundingBox: rect)
                    unorderedPredictions.append(prediction)
                }
            }
            
            //Array to store final predictions (after post-processing)
            var predictions: [Prediction] = []
            let orderedPredictions = unorderedPredictions.sorted { $0.confidence > $1.confidence }
            var keep = [Bool](repeating: true, count: orderedPredictions.count)
            for i in 0..<orderedPredictions.count {
                if keep[i] {
                    predictions.append(orderedPredictions[i])
                    let bbox1 = orderedPredictions[i].boundingBox
                    for j in (i+1)..<orderedPredictions.count {
                        if keep[j] {
                            let bbox2 = orderedPredictions[j].boundingBox
                            if self.IoU(bbox1, bbox2) > 0.01 {
                                keep[j] = false
                            }
                        }
                    }
                }
            }
            
            var s = ""
            
            var topPrediction : Prediction = predictions[0]
            
            for p in predictions {
                s.append("\(labels[p.labelIndex]) + height \(p.boundingBox.height) + width \(p.boundingBox.width) + midX \(p.boundingBox.midX) + midY \(p.boundingBox.midY) /n")
                if (p.confidence > topPrediction.confidence){
                    topPrediction = p
                }
//                print(p.labelIndex)
//                print(p.confidence)
//                print(labels[p.labelIndex])
//                print("rect")
//                print("height \(p.boundingBox.height)")
//                print("width \(p.boundingBox.width)")
//                print("midX \(p.boundingBox.midX)")
//                print("midY \(p.boundingBox.midY)")
            }
            

            
        }
    }
    
    // MARK: - HIDE STATUS BAR
    override var prefersStatusBarHidden : Bool { return true }
}
