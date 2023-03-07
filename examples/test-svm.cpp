#include <dlib/svm_threaded.h>
#include <dlib/string.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>


#include <iostream>
#include <fstream>


using namespace std;
using namespace dlib;

int main(int argc, char ** argv){
    try{
        typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;
        command_line_parser parser;
        parser.add_option("u", "Upsample each input image <arg> times. Each upsampling quadruples the number of pixels in the image (default: 0).", 1);
        parser.parse(argc, argv);
        const int upsample_amount = get_option(parser,"u", 0);

        // load a previously trained object detector and try it out on some data
        ifstream fin("masked.svm", ios::binary);
        if (!fin)
        {
            cout << "Can't find a trained object detector file object_detector.svm. " << endl;
            cout << "You need to train one using the -t option." << endl;
            cout << "\nTry the -h option for more information." << endl;
            return EXIT_FAILURE;

        }
        object_detector<image_scanner_type> detector;
        deserialize(detector, fin);

        dlib::array<array2d<unsigned char> > images;
        // Check if the command line argument is an XML file
        if (tolower(right_substr(parser[0],".")) == "xml")
        {
            std::vector<std::vector<rectangle> > object_locations, ignore;
            cout << "Loading image dataset from metadata file " << parser[0] << endl;
            ignore = load_image_dataset(images, object_locations, parser[0]);
            cout << "Number of images loaded: " << images.size() << endl;

            // Upsample images if the user asked us to do that.
            for (unsigned long i = 0; i < upsample_amount; ++i)
                upsample_image_dataset<pyramid_down<2> >(images, object_locations, ignore);

        
        
            cout << "Testing detector on data..." << endl;
            cout << "Results (precision,recall,AP): " << test_object_detection_function(detector, images, object_locations, ignore) << endl;
            return EXIT_SUCCESS;
        
        }
        else
        {
            // In this case, the user should have given some image files.  So just
            // load them.
            images.resize(parser.number_of_arguments());
            for (unsigned long i = 0; i < images.size(); ++i)
                load_image(images[i], parser[i]);

            // Upsample images if the user asked us to do that.
            for (unsigned long i = 0; i < upsample_amount; ++i)
            {
                for (unsigned long j = 0; j < images.size(); ++j)
                    pyramid_up(images[j]);
            }
        }


        // Test the detector on the images we loaded and display the results
        // in a window.
        image_window win;
        for (unsigned long i = 0; i < images.size(); ++i)
        {
            // Run the detector on images[i] 
            const std::vector<rectangle> rects = detector(images[i]);
            cout << "Number of detections: "<< rects.size() << endl;

            // Put the image and detections into the window.
            win.clear_overlay();
            win.set_image(images[i]);
            win.add_overlay(rects, rgb_pixel(255,0,0));

            cout << "Hit enter to see the next image.";
            cin.get();
        }


    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
        cout << "\nTry the -h option for more information." << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;

}
