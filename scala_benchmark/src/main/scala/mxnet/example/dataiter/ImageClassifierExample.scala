/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package mxnet.example.dataiter

import org.apache.mxnet.Shape
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory
import org.apache.mxnet.{DType, DataDesc}
import org.apache.mxnet.infer._

import scala.collection.JavaConverters._
import java.io.File
import scala.collection.mutable.ListBuffer

/**
  * Example showing usage of Infer package to do inference on resnet-152 model
  * Follow instructions in README.md to run this example.
  */
object ImageClassifierExample {
  private val logger = LoggerFactory.getLogger(classOf[ImageClassifierExample])

  def runInferenceOnSingleImage(modelPathPrefix: String, inputImagePath: String, numRun: Int):
  Unit = {

    val dType = DType.Float32
    val inputShape = Shape(1, 3, 224, 224)

    val inputDescriptor = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))

    // Create object of ImageClassifier class
    val imgClassifier: ImageClassifier = new
        ImageClassifier(modelPathPrefix, inputDescriptor)

    // Loading single image from file and getting BufferedImage
    val img = ImageClassifier.loadImageFromFile(inputImagePath)

    for (i <- 1 to numRun) {
      val startTimeSingle = System.nanoTime
      // Running inference on single image
      val output = imgClassifier.classifyImage(img, Some(5))

      val estimatedTimeSingle = (System.nanoTime() - startTimeSingle)
      printf("Iteration: %d, Time: %d \n",i, estimatedTimeSingle)

    }
  }

  def runInferenceOnBatchOfImage(modelPathPrefix: String, inputImageDir: String, numRun: Int):
  Unit = {
    val dType = DType.Float32
    val inputShape = Shape(1, 3, 224, 224)

    val inputDescriptor = IndexedSeq(DataDesc("data", inputShape, dType, "NCHW"))

    // Create object of ImageClassifier class
    val imgClassifier: ImageClassifier = new
        ImageClassifier(modelPathPrefix, inputDescriptor)

    // Loading batch of images from the directory path
    val batchFiles = generateBatches(inputImageDir, 20)
    var outputList = IndexedSeq[IndexedSeq[(String, Float)]]()

for (i <- 1 to numRun) {
    val startTimeSingle = System.nanoTime
    for (batchFile <- batchFiles) {
      val imgList = ImageClassifier.loadInputBatch(batchFile)
      // Running inference on batch of images loaded in previous step
      outputList ++= imgClassifier.classifyImageBatch(imgList, Some(5))
    }
    val estimatedTimeSingle = (System.nanoTime() - startTimeSingle)
      printf("Iteration: %d, Time: %d \n",i, estimatedTimeSingle)
  }
  }

  def generateBatches(inputImageDirPath: String, batchSize: Int = 100): List[List[String]] = {
    val dir = new File(inputImageDirPath)
    require(dir.exists && dir.isDirectory,
      "input image directory: %s not found".format(inputImageDirPath))
    val output = ListBuffer[List[String]]()
    var batch = ListBuffer[String]()
    for (imgFile: File <- dir.listFiles()){
      batch += imgFile.getPath
      if (batch.length == batchSize) {
        output += batch.toList
        batch = ListBuffer[String]()
      }
    }
    if (batch.length > 0) {
      output += batch.toList
    }
    output.toList
  }

  def main(args: Array[String]): Unit = {
    val inst = new ImageClassifierExample
    val parser: CmdLineParser = new CmdLineParser(inst)
      parser.parseArgument(args.toList.asJava)

      val modelPathPrefix = if (inst.modelPathPrefix == null) System.getenv("MXNET_DATA_DIR")
      else inst.modelPathPrefix

      val inputImagePath = if (inst.inputImagePath == null) System.getenv("MXNET_DATA_DIR")
      else inst.inputImagePath

      val inputImageDir = if (inst.inputImageDir == null) System.getenv("MXNET_DATA_DIR")
      else inst.inputImageDir

      val numRun = inst.numRun.toString().toInt

//      runInferenceOnSingleImage(modelPathPrefix, inputImagePath, numRun)
       runInferenceOnBatchOfImage(modelPathPrefix, inputImageDir, numRun)

  }
}

class ImageClassifierExample {
  @Option(name = "--model-path-prefix", usage = "the input model directory")
  private val modelPathPrefix: String = "/Users/roshanin/Downloads/resnet/resnet-152"
  @Option(name = "--input-image", usage = "the input image")
  private val inputImagePath: String = "/Users/roshanin/Downloads/kitten.jpg"
  @Option(name = "--input-dir", usage = "the input batch of images directory")
  private val inputImageDir: String = "/Users/roshanin/Downloads/images/"
  @Option(name = "--num-run", usage = "number of times to run inference")
  private val numRun: Int = 1
}
