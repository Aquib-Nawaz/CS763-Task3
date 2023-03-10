#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>


using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------
template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon3  = relu<bn_con<con3<32,SUBNET>>>;
using net_type  = loss_mmod<con<1,6,6,1,1,rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------


int main(int argc, char** argv) try
{
    if (argc == 1)
    {
        cout << "Call this program like this:" << endl;
        cout << "./test-dnn ann.dat faces/*.jpg" << endl;
        cout << "\nYou can get the mmod_human_face_detector.dat file from:\n";
        cout << "running train-dnn file" << endl;
        return 0;
    }


    net_type net; 
    net.subnet().layer_details().set_num_filters(4);
    deserialize(argv[1]) >> net;  

    image_window win;
    for (int i = 2; i < argc; ++i)
    {
        matrix<rgb_pixel> img;
        load_image(img, argv[i]);

        // Upsampling the image will allow us to detect smaller faces but will cause the
        // program to use more RAM and run longer.
        while(img.size() < 1800*1800)
            pyramid_up(img);

        // Note that you can process a bunch of images in a std::vector at once and it runs
        // much faster, since this will form mini-batches of images and therefore get
        // better parallelism out of your GPU hardware.  However, all the images must be
        // the same size.  To avoid this requirement on images being the same size we
        // process them individually in this example.
        auto dets = net(img);
        win.clear_overlay();
        win.set_image(img);
        for (auto&& d : dets)
            win.add_overlay(d);

        cout << "Hit enter to process the next image." << endl;
        cin.get();
    }
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}


