#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <dlib/data_io.h>
#include <dlib/string.h>
#include <fstream>
#include <dlib/svm_threaded.h>
#include <iostream>


using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    try
    {
        
        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./test-shape sp.dat faces/*.jpg" << endl;
            cout << "\nYou can get the sp.dat file from running train-svm.cpp\n";
            
            return 0;
        }

        
	dlib::array<array2d<unsigned char> > images;
        std::vector<std::vector<rectangle> > object_locations, ignore;
            cout << "Loading image dataset from metadata file " << argv[2] << endl;
            ignore = load_image_dataset(images, object_locations, argv[2]);
            cout << "Number of images loaded: " << images.size() << endl;
	using namespace dlib::image_dataset_metadata;
	dataset data;
	load_image_dataset_metadata(data, argv[2]);
	cout << "Number of images loaded: " << data.images.size() << endl;	
//save annotations
        
        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        shape_predictor sp;
        deserialize(argv[1]) >> sp;
        // Loop over all the images provided on the command line.
        const rgb_pixel color(0,255,0);
        for (int i = 0; i < images.size(); ++i)
        {
            
    		
            // Now tell the face detector to give us a list of bounding boxes
            // around all the faces in the image.
     
            // Now we will go ask the shape_predictor to tell us the pose of
            // each face we detected.
	    std::vector<rectangle> dets = object_locations[i];
            for (unsigned long j = 0; j < dets.size(); ++j)
            {
                full_object_detection shape = sp(images[i], dets[j]);
                
                // You get the idea, you can get all the face part locations if
                // you want them.  Here we just store them in shapes so we can
                // put them on the screen.
		
                for (int k=0; k<=9; k++){
                string zero = "0";
		zero+=to_string(k);
    data.images[i].boxes[j].parts[zero]=shape.part(k);
                }
		for (int k=10;k<shape.num_parts(); k++){
		data.images[i].boxes[j].parts[to_string(k)]=shape.part(k);
		}
                // 
                
            }

            
        }
	save_image_dataset_metadata(data, "../data/landmarks.xml");
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

