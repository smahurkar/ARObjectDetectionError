/*
See LICENSE folder for this sample’s licensing information.

Abstract:
View controller for selecting images and applying Vision + Core ML processing.
*/

import UIKit
import CoreML
import Vision
import ImageIO

class ImageClassificationViewController: UIViewController {
    // MARK: - IBOutlets
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var cameraButton: UIBarButtonItem!
    @IBOutlet weak var classificationLabel: UILabel!

    
    //struct to handle multi arrays
    struct Prediction {
        let labelIndex: Int
        let confidence: Float
        let boundingBox: CGRect
    }
    
    // MARK: - Image Classification
    
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
                self?.processClassifications(for: request, error: error)
            })
            request.imageCropAndScaleOption = .scaleFill
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    
    /// - Tag: PerformRequests
    func updateClassifications(for image: UIImage) {
        classificationLabel.text = "Classifying..."
        
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
    }
    
    /// Updates the UI with the results of the classification.
    /// - Tag: ProcessClassifications
    func processClassifications(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            let results = request.results as! [VNCoreMLFeatureValueObservation]

            let mlmodel = MyCustomObjectDetector()
            
            let userDefined: [String: String] = mlmodel.model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey]! as! [String : String]
            let labels = userDefined["classes"]!.components(separatedBy: ",")
            let nmsThreshold = Float(userDefined["non_maximum_suppression_threshold"]!) ?? 0.01

            let coordinates = results[0].featureValue.multiArrayValue!
            let confidence = results[1].featureValue.multiArrayValue!
            
            print("confidence is")
            print(confidence)
            
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
            
            print("reaching predictions stage")
            
            print(predictions.count)
            
            for p in predictions {
                print(p.labelIndex)
                print(p.confidence)
                print(labels[p.labelIndex])
                print("rect")
                print("height \(p.boundingBox.height)")
                print("width \(p.boundingBox.width)")
                print("midX \(p.boundingBox.midX)")
                print("midY \(p.boundingBox.midY)")
                self.classificationLabel.text = labels[p.labelIndex]
            }
        
        }
    }
    
    // MARK: - Photo Actions
    
    
    public func IoU(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        let union = a.union(b)
        return Float((intersection.width * intersection.height) / (union.width * union.height))
    }
    
    @IBAction func takePicture() {
        // Show options for the source picker only if the camera is available.
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            presentPhotoPicker(sourceType: .photoLibrary)
            return
        }
        
        let photoSourcePicker = UIAlertController()
        let takePhoto = UIAlertAction(title: "Take Photo", style: .default) { [unowned self] _ in
            self.presentPhotoPicker(sourceType: .camera)
        }
        let choosePhoto = UIAlertAction(title: "Choose Photo", style: .default) { [unowned self] _ in
            self.presentPhotoPicker(sourceType: .photoLibrary)
        }
        
        photoSourcePicker.addAction(takePhoto)
        photoSourcePicker.addAction(choosePhoto)
        photoSourcePicker.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
        
        present(photoSourcePicker, animated: true)
    }
    
    func presentPhotoPicker(sourceType: UIImagePickerControllerSourceType) {
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = sourceType
        present(picker, animated: true)
    }
}

extension ImageClassificationViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    // MARK: - Handling Image Picker Selection

    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String: Any]) {
        picker.dismiss(animated: true)
        
        // We always expect `imagePickerController(:didFinishPickingMediaWithInfo:)` to supply the original image.
        let image = info[UIImagePickerControllerOriginalImage] as! UIImage
        imageView.image = image
        updateClassifications(for: image)
    }
}
