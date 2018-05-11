/*
 * Wrapper on OpenPose API
 * OpenPose Repo: https://github.com/CMU-Perceptual-Computing-Lab/openpose
 *
 * Code has been adapted from OpenPose project examples.
 */

#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#define USE_CAFFE

#include <gflags/gflags.h> // DEFINE_bool, DEFINE_int32, DEFINE_int64, DEFINE_uint64, DEFINE_double, DEFINE_string
#include <glog/logging.h> // google::InitGoogleLogging, CHECK, CHECK_EQ, LOG, VLOG, ...

#include <stdio.h>
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>


#include <pyboostcvconverter/pyboostcvconverter.hpp>

#include <boost/python.hpp>


#include <iostream>

 using namespace boost::python;


DEFINE_int32(logging_level,             3,              "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while 255 will not output any."
                                                        " Current OpenPose library messages are in the range 0-4: 1 for low priority messages and 4 for important ones.");
// OpenPose
DEFINE_string(model_pose,               "COCO",         "Model to be used (e.g. COCO, MPI, MPI_4_layers).");
DEFINE_string(model_folder,             "models/",      "Folder where the pose models (COCO and MPI) are located.");
DEFINE_string(net_resolution,           "656x368",      "Multiples of 16.");
DEFINE_string(resolution,               "1280x720",     "The image resolution (display). Use \"-1x-1\" to force the program to use the default images resolution.");
DEFINE_int32(num_gpu_start,             0,              "GPU device start number.");
DEFINE_double(scale_gap,                0.3,            "Scale gap between scales. No effect unless num_scales>1. Initial scale is always 1. If you want to change the initial scale, "
                                                        "you actually want to multiply the `net_resolution` by your desired initial scale.");
DEFINE_int32(num_scales,                1,              "Number of scales to average.");
// OpenPose Rendering
DEFINE_double(alpha_pose,               0.6,            "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will hide it.");

using namespace std;

cv::Size outputSize;
cv::Size netInputSize;
cv::Size netOutputSize;
op::PoseModel poseModel;
op::CvMatToOpInput * cvMatToOpInput;
op::CvMatToOpOutput * cvMatToOpOutput;
cv::Size windowedSize;

op::PoseRenderer * poseRenderer;

op::OpOutputToCvMat * opOutputToCvMat;

op::PoseExtractorCaffe * poseExtractorCaffe;
op::FrameDisplayer * frameDisplayer;

bool renderOutputs = false;

cv::Mat outputImage;



void error(const char *msg)
{
	perror(msg);
	exit(1);
}



op::PoseModel gflagToPoseModel(const std::string& poseModeString)
{
	op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
	if (poseModeString == "COCO")
		return op::PoseModel::COCO_18;
	else if (poseModeString == "MPI")
		return op::PoseModel::MPI_15;
	else if (poseModeString == "MPI_4_layers")
		return op::PoseModel::MPI_15_4;
	else
	{
		op::error("String does not correspond to any model (COCO, MPI, MPI_4_layers)", __LINE__, __FUNCTION__, __FILE__);
		return op::PoseModel::COCO_18;
	}
}

// Google flags into program variables
std::tuple<cv::Size, cv::Size, cv::Size, op::PoseModel> gflagsToOpParameters(int netWidth, int netHeight)
{
	op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
	// outputSize
	cv::Size outputSize;
	auto nRead = sscanf(FLAGS_resolution.c_str(), "%dx%d", &outputSize.width, &outputSize.height);
	op::checkE(nRead, 2, "Error, resolution format (" +  FLAGS_resolution + ") invalid, should be e.g., 960x540 ", __LINE__, __FUNCTION__, __FILE__);
	// netInputSize
	cv::Size netInputSize;
	nRead = sscanf(FLAGS_net_resolution.c_str(), "%dx%d", &netInputSize.width, &netInputSize.height);
	netInputSize.width = netWidth;
	netInputSize.height = netHeight;
	op::checkE(nRead, 2, "Error, net resolution format (" +  FLAGS_net_resolution + ") invalid, should be e.g., 656x368 (multiples of 16)", __LINE__, __FUNCTION__, __FILE__);
	// netOutputSize
	const auto netOutputSize = netInputSize;
	// poseModel
	const auto poseModel = gflagToPoseModel(FLAGS_model_pose);
	// Check no contradictory flags enabled
	if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
		op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
	if (FLAGS_scale_gap <= 0. && FLAGS_num_scales > 1)
		op::error("Uncompatible flag configuration: scale_gap must be greater than 0 or num_scales = 1.", __LINE__, __FUNCTION__, __FILE__);
	// Logging and return result
	op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
	return std::make_tuple(outputSize, netInputSize, netOutputSize, poseModel);
}

/**
 * Initializes API
 * @param renderOutputs Should OP Renderer be used to show outputs
 * @param netWidth Width of net
 * @param netHeight Height of net
 */
void setup(bool renderOutputs, int netWidth, int netHeight){
	Py_BEGIN_ALLOW_THREADS
	op::log("Here we got started!!");
	if(renderOutputs){
		op::log("Outputs will be rendered!");
	}
	else{
		op::log("Outputs will NOT be rendered!");
	}


	op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
	op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);

	op::log(netWidth);
	op::log(netHeight);

	std::tie(outputSize, netInputSize, netOutputSize, poseModel) = gflagsToOpParameters(netWidth,netHeight);

	// Step 3 - Initialize all required classes
	cvMatToOpInput = new op::CvMatToOpInput(netInputSize, FLAGS_num_scales, (float)FLAGS_scale_gap);
	cvMatToOpOutput = new op::CvMatToOpOutput(outputSize);

	poseExtractorCaffe = new op::PoseExtractorCaffe(netInputSize, netOutputSize, outputSize, FLAGS_num_scales, (float)FLAGS_scale_gap, poseModel,
											  FLAGS_model_folder, FLAGS_num_gpu_start);

	if(renderOutputs)
		poseRenderer = new op::PoseRenderer (netOutputSize, outputSize, poseModel, nullptr, (float)FLAGS_alpha_pose);

	opOutputToCvMat = new op::OpOutputToCvMat(outputSize);
	windowedSize = outputSize;

	if(renderOutputs)
		frameDisplayer = new op::FrameDisplayer(windowedSize, "Custom Code");

	 // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
	poseExtractorCaffe->initializationOnThread();

	if(renderOutputs)
		poseRenderer->initializationOnThread();

	op::log("setup ended!");

	Py_END_ALLOW_THREADS

}


/**
 * Estimate post of input image
 * @param inputImage Input image as an OpenCV Matrix
 * @return
 */
cv::Mat estimatePoseMat(cv::Mat inputImage)
{
	if(inputImage.empty())
		op::log("Empty Image");
	// Step 2 - Format input image to OpenPose input and output formats
	const auto netInputArray = cvMatToOpInput->format(inputImage);
	//op::log("Formatted!");
	double scaleInputToOutput;
	op::Array<float> outputArray;
	std::tie(scaleInputToOutput, outputArray) = cvMatToOpOutput->format(inputImage);

	// Step 3 - Estimate poseKeyPoints
	poseExtractorCaffe->forwardPass(netInputArray, inputImage.size());
	const auto poseKeyPoints = poseExtractorCaffe->getPoseKeyPoints();

	//op::log("Pose Estimated");


	int count = poseKeyPoints.getSize(0);


	std::ostringstream resultBuilder;

	if(renderOutputs)
	{
		poseRenderer->renderPose(outputArray, poseKeyPoints);
	}
	outputImage = opOutputToCvMat->formatToCvMat(outputArray);

	if(count == 0)
	{
		return cv::Mat();
	}

	auto outputResult = poseKeyPoints.getConstCvMat();

	//op::log("Result converted to matrix");


	return outputResult;
}

/**
 * Retrieve output image of OpenPose Renderer
 * @return
 */
PyObject * getOutputImage()
{
	PyObject *ret = pbcvt::fromMatToNDArray(outputImage);
	return ret;
}

/**
 * Detect persons and their poses using OpenPose Detector
 * @param frame
 * @return
 */
PyObject *detect(PyObject *frame)
{
	PyObject *ret = 0;
	cv::Mat result;

	//op::log("Detect Called");
	cv::Mat frameMat;;
	frameMat = pbcvt::fromNDArrayToMat(frame);
	//op::log("Frame Converted");

	Py_BEGIN_ALLOW_THREADS

	result =  estimatePoseMat(frameMat);
	//op::log("Estimated");

	Py_END_ALLOW_THREADS

	ret = pbcvt::fromMatToNDArray(result);

	return ret;

}

/**
 * Retrieve output width
 * @return
 */
int getOutputWidth()
{
	return outputImage.size[1];
}

/**
 * Retrieve output height
 * @return
 */
int getOutputHeight()
{
	return outputImage.size[0];
}

// Initializations

#if (PY_VERSION_HEX >= 0x03000000)

    static void *init_ar() {
#else
        static void init_ar(){
#endif
        Py_Initialize();
        PyEval_InitThreads();

        import_array();
        return NUMPY_IMPORT_ARRAY_RETVAL;
    }


/**
 * Init Module
 */
BOOST_PYTHON_MODULE(libOpenPersonDetectorAPI)
{
	init_ar();

	//initialize converters
	to_python_converter<cv::Mat,
			pbcvt::matToNDArrayBoostConverter>();
	pbcvt::matFromNDArrayBoostConverter();

    // Initialize endpoints
    def("detect", detect);
    def("getOutputImage", getOutputImage);
    def("getOutputWidth", getOutputWidth);
    def("getOutputHeight", getOutputHeight);
    def("setup", setup);

}

