#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include <sys/time.h>
#include <stdlib.h>
//#define _GNU_SOURCE
#include <string.h>
//#include <stdio.h>
//#include <libgen.h>
//#include <unistd.h>

}
#define CLS_NUM 21
#ifdef OPENCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


extern "C" image ipl_to_image(IplImage* src);
extern "C" void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);
extern "C" void draw_yolo(image im, int num, float thresh, box *boxes, float **probs);
extern "C" void crop_detection_coords(image im, int num, float thresh, box *boxes, float **probs, char **names, image *labels, int classes, int target_class, int *left, int *right, int *top, int *bot);
extern "C" image crop_image(image im, int dx, int dy, int w, int h);
extern "C" char *voc_names[];
extern "C" image voc_labels[];
//extern "C" char *basename(char const *path);
//extern "C" char *concat(char s1, char s2);
static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image det  ;
static image det_s;
static image disp ;
static int left, right, top, bot;
static cv::VideoCapture cap;
static float fps = 0;
static float demo_thresh = 0;
static int end_flag = 0;

void *fetch_in_thread(void *ptr)
{
    cv::Mat frame_m;
    end_flag = 1-cap.read(frame_m);
    if (end_flag==0){
    IplImage frame = frame_m;
    in = ipl_to_image(&frame);
    
    rgbgr_image(in);
    in_s = resize_image(in, net.w, net.h);
    printf("got here 20");
    }

    return 0;
}

void *detect_in_thread(void *ptr)
{   
    if(gpu_index >= 0){
        cudaError_t status = cudaSetDevice(gpu_index);
        check_error(status);
        fprintf(stdout, "gpu: %i \n", gpu_index);
    }

    if (end_flag==0){
    float nms = .4;
    printf("got here s-1\n");
    detection_layer l = net.layers[net.n-1];
    float *X = det_s.data;
    printf("got here s0\n");
    float *predictions = network_predict(net, X);
    free_image(det_s);
    printf("got here s1\n");
    convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
    printf("got here s2\n");
    if (nms > 0) {
    do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    printf("got here s3\n");
    crop_detection_coords(det, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, CLS_NUM, 20, &left, &right, &top, &bot);
    printf("got here s4\n");
    } else {
    left = 0;
    right = 0;
    top = 0;
    bot = 0;
    }
    
    printf("\033[2J");
    printf("\033[1;1H");
    printf("\nFPS:%.0f\n",fps);
    printf("got here -5");  
    printf("Objects:\n\n");
    printf("got here -2");
    }
    return 0;
}

/*char *basename(char const *path)
{
        char *s = strrchr(path, '/');
        if(s==NULL) {
                return strdup(path);
        } else {
                return strdup(s + 1);
        }
}*/

extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename)
{
    demo_thresh = thresh;
    printf("YOLO demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);

    if(filename){
        cap.open(filename);
    }else{
	cv::VideoCapture cam(cam_index);
    	cap = cam;
        cap.open(cam_index);
    }


    if(!cap.isOpened()) error("Couldn't connect to webcam.\n");

    detection_layer l = net.layers[net.n-1];
    int j;
    
    //printf("got here -4");  
    boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
    probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));

    pthread_t fetch_thread;
    pthread_t detect_thread;

    fetch_in_thread(0);
    det = in;
    det_s = in_s;
    
    fetch_in_thread(0);
    detect_in_thread(0);
    disp = det;
    det = in;
    det_s = in_s;

    int cnt = 0;
    
    while(end_flag==0){
        //printf("got here-1"); 
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
        /*if (end_flag == 0){
	
        //disp2=crop_image(disp, left, top, right-left, bot-top);
	//save_image2(disp2, "tmp/", cnt);
        //show_image(disp2, "YOLO");
        }*/
        //printf("got here0");

        //char *name = malloc(strlen(filename));
        char *name=strdup(filename);
        char savename[512];
        
        sprintf(savename, "tmp/%scoords.txt", basename(name));
        //printf("got here1\n");
        //strcat(savedir, basename(savename));
        //printf("got here2\n");
        if (cnt%6==0){
        save_crop_coords(savename, left, right, top, bot);
        } 
        save_crop_coords(savename, left, right, top, bot);
        printf("got here3\n");
	cnt = cnt+1;
        free_image(disp);
        //printf("got here1\n");
        cvWaitKey(1);
        pthread_join(fetch_thread, 0);
        //printf("got here1.5\n");
        pthread_join(detect_thread, 0);
        //printf("got here2");
        disp  = det;
        det   = in;
        det_s = in_s;

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        printf("got here3");
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
}
#else
extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index){
    fprintf(stderr, "YOLO demo needs OpenCV for webcam images.\n");
}
#endif

