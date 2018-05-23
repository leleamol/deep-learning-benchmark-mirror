package mxnet.example.dataiter

import java.awt.image.BufferedImage
import java.io.File
import java.net.URL

import scala.io.Source
import javax.imageio.ImageIO

import org.apache.mxnet._
import org.apache.mxnet.module.{FitParams, Module}
import org.apache.mxnet.io.NDArrayIter
import org.kohsuke.args4j.{CmdLineParser, Option}
import org.slf4j.LoggerFactory

import scala.collection.immutable.ListMap
import scala.collection.mutable.ListBuffer
import scala.collection.JavaConverters._

import sys.process._

object BenchmarkResnet {

  private val logger = LoggerFactory.getLogger(classOf[BenchmarkResnet])

  def getScaledImage(img: BufferedImage, newWidth: Int, newHeight: Int): BufferedImage = {
    val resizedImage = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB)
    val g = resizedImage.createGraphics()
    g.drawImage(img, 0, 0, newWidth, newHeight, null)
    g.dispose()

    resizedImage
  }

  def getImage(resizedImage: BufferedImage, inputImageShape: Shape): NDArray = {
    // Get height and width of the image
    val w = resizedImage.getWidth()
    val h = resizedImage.getHeight()

    // get an array of integer pixels in the default RGB color mode
    val pixels = resizedImage.getRGB(0, 0, w, h, null, 0, w)

    // 3 times height and width for R,G,B channels
    val result = new Array[Float](3 * h * w)

    var row = 0
    // copy pixels to array vertically
    while (row < h) {
      var col = 0
      // copy pixels to array horizontally
      while (col < w) {
        val rgb = pixels(row * w + col)
        // getting red color
        result(0 * h * w + row * w + col) = (rgb >> 16) & 0xFF
        // getting green color
        result(1 * h * w + row * w + col) = (rgb >> 8) & 0xFF
        // getting blue color
        result(2 * h * w + row * w + col) = rgb & 0xFF
        col += 1
      }
      row += 1
    }

    // creating NDArray according to the input shape
    val pixelsArray = NDArray.array(result, shape = inputImageShape)
    pixelsArray
  }

  def readSynsetFile(synsetFilePath: String): List[String] = {
    val f = Source.fromFile(synsetFilePath)
    try {
      f.getLines().toList
    } finally {
      f.close
    }
  }

  def runInference(inputDescriptors: IndexedSeq[DataDesc], modelPathPrefix: String, inputImagePath: String,
                   synsetPath: String, numRun: Int): Unit = {

    val mod = Module.loadCheckpoint(modelPathPrefix, 0)

    mod.bind(dataShapes = inputDescriptors)

    val img = ImageIO.read(new File(inputImagePath))
val labelsList = readSynsetFile(synsetPath)

    for (i <- 1 to numRun) {
      val startTimeSingle = System.nanoTime

    val resizedImg = getScaledImage(img, inputDescriptors(0).shape(2), inputDescriptors(0).shape(3))
    val pixelsNdarray = getImage(resizedImg, inputDescriptors(0).shape)
    img.flush()
    resizedImg.flush()

    mod.predict(new NDArrayIter(IndexedSeq(pixelsNdarray)))
    val predictResult = mod.getOutputs()(0)

    val prob = predictResult(0)

    val topProbs = NDArray.argsort(prob).toArray.reverse.slice(0,5)
     
    val estimatedTimeSingle = (System.nanoTime() - startTimeSingle)
    printf("Iteration: %d, Time: %d \n",i, estimatedTimeSingle)

    }

  }

  def main(args: Array[String]): Unit = {
    val inst = new BenchmarkResnet
    val parser: CmdLineParser = new CmdLineParser(inst)
    try {    
    parser.parseArgument(args.toList.asJava)

    val dType = DType.Float32
    val dShape = Shape(1, 3, 224, 224)
    val inputDescriptors = IndexedSeq(DataDesc("data", dShape, dType, "NCHW"))

    val modelPathPrefix = if (inst.modelPathPrefix == null) System.getenv("MXNET_DATA_DIR")
    else inst.modelPathPrefix

    val inputImagePath = if (inst.inputImagePath == null) System.getenv("MXNET_DATA_DIR")
    else inst.inputImagePath

    val synsetPath = if (inst.synsetPath == null) System.getenv("MXNET_DATA_DIR")
    else inst.synsetPath

    val numRun = inst.numRun.toString().toInt

      runInference(inputDescriptors, modelPathPrefix, inputImagePath, synsetPath, numRun)
  
    } catch {
      case ex: Exception => {
        logger.error(ex.getMessage, ex)
        parser.printUsage(System.err)
        sys.exit(1)
      }
    }
  }
}

class BenchmarkResnet {
  @Option(name = "--model-path-prefix", usage = "the input model directory")
  private val modelPathPrefix: String = "/Users/roshanin/Downloads/resnet/resnet-152"
  @Option(name = "--input-image", usage = "the input image")
  private val inputImagePath: String = "/Users/roshanin/Downloads/kitten.jpg"
  @Option(name = "--input-dir", usage = "the input batch of images directory")
  private val inputImageDir: String = "/images/"
  @Option(name = "--synset", usage = "synset file path")
  private val synsetPath: String = "/Users/roshanin/Downloads/resnet/synset.txt"
  @Option(name = "--num-run", usage = "number of times to run inference")
  private val numRun: Int = 1
}
