

#include <iostream>
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/gui_widgets.h>

using namespace std;
using namespace dlib;


template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con3  = con<num_filters,3,3,1,1,SUBNET>;
template <typename SUBNET> using downsampler  = relu<bn_con<con5d<32, relu<bn_con<con5d<32, relu<bn_con<con5d<32,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon3  = relu<bn_con<con3<32,SUBNET>>>;
using net_type  = loss_mmod<con<1,6,6,1,1,rcon3<rcon3<rcon3<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    // In this example we are going to train a face detector based on the
    // small faces dataset in the examples/faces directory.  So the first
    // thing we do is load that dataset.  This means you need to supply the
    // path to this faces folder as a command line argument so we will know
    // where it is.
    if (argc != 2)
    {
        cout << "Give the path to the examples/faces directory as the argument to this" << endl;
        cout << "program.  For example, if you are in the examples folder then execute " << endl;
        cout << "this program by running: " << endl;
        cout << "   ./dnn_mmod_ex faces" << endl;
        cout << endl;
        return 0;
    }
    const std::string faces_directory = argv[1];
    
    std::vector<matrix<rgb_pixel>> images_train, images_test;
    std::vector<std::vector<mmod_rect>> face_boxes_train, face_boxes_test;

    load_image_dataset(images_train, face_boxes_train, faces_directory+"/training.xml");
    load_image_dataset(images_test, face_boxes_test, faces_directory+"/training.xml");


    cout << "num training images: " << images_train.size() << endl;
    cout << "num testing images:  " << images_test.size() << endl;


    // The MMOD algorithm has some options you can set to control its behavior.  However,
    // you can also call the constructor with your training annotations and a "target
    // object size" and it will automatically configure itself in a reasonable way for your
    // problem.  Here we are saying that faces are still recognizably faces when they are
    // 40x40 pixels in size.  You should generally pick the smallest size where this is
    // true.  Based on this information the mmod_options constructor will automatically
    // pick a good sliding window width and height.  It will also automatically set the
    // non-max-suppression parameters to something reasonable.  For further details see the
    // mmod_options documentation.
    mmod_options options(face_boxes_train, 40,40);
    // The detector will automatically decide to use multiple sliding windows if needed.
    // For the face data, only one is needed however.
    cout << "num detector windows: "<< options.detector_windows.size() << endl;
    for (auto& w : options.detector_windows)
        cout << "detector window width by height: " << w.width << " x " << w.height << endl;
    cout << "overlap NMS IOU thresh:             " << options.overlaps_nms.get_iou_thresh() << endl;
    cout << "overlap NMS percent covered thresh: " << options.overlaps_nms.get_percent_covered_thresh() << endl;

    // Now we are ready to create our network and trainer.  
    net_type net(options);
    // The MMOD loss requires that the number of filters in the final network layer equal
    // options.detector_windows.size().  So we set that here as well.
    net.subnet().layer_details().set_num_filters(options.detector_windows.size());
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.1);
    trainer.be_verbose();
    trainer.set_synchronization_file("temp_model_0x.dat", std::chrono::minutes(5));
    trainer.set_iterations_without_progress_threshold(40);


    // Now let's train the network.  We are going to use mini-batches of 150
    // images.   The images are random crops from our training set (see
    // random_cropper_ex.cpp for a discussion of the random_cropper). 
    std::vector<matrix<rgb_pixel>> mini_batch_samples;
    std::vector<std::vector<mmod_rect>> mini_batch_labels; 
    random_cropper cropper;
    cropper.set_chip_dims(200, 200);
    // Usually you want to give the cropper whatever min sizes you passed to the
    // mmod_options constructor, which is what we do here.
    cropper.set_min_object_size(40,40);
    dlib::rand rnd;
    // Run the trainer until the learning rate gets small.  This will probably take several
    // hours.
    while(trainer.get_learning_rate() >= 1e-4)
    {
        cropper(150, images_train, face_boxes_train, mini_batch_samples, mini_batch_labels);
        // We can also randomly jitter the colors and that often helps a detector
        // generalize better to new images.
        for (auto&& img : mini_batch_samples)
            disturb_colors(img, rnd);

        trainer.train_one_step(mini_batch_samples, mini_batch_labels);
    }
    // wait for training threads to stop
    trainer.get_net();
    cout << "done training" << endl;

    // Save the network to disk
    net.clean();
    serialize("ann.dat") << net;


    // Now that we have a face detector we can test it.  The first statement tests it
    // on the training data.  It will print the precision, recall, and then average precision.
    // This statement should indicate that the network works perfectly on the
    // training data.
    cout << "training results: " << test_object_detection_function(net, images_train, face_boxes_train) << endl;
    // However, to get an idea if it really worked without overfitting we need to run
    // it on images it wasn't trained on.  The next line does this.   Happily,
    // this statement indicates that the detector finds most of the faces in the
    // testing data.
    cout << "testing results:  " << test_object_detection_function(net, images_test, face_boxes_test) << endl;


    // If you are running many experiments, it's also useful to log the settings used
    // during the training experiment.  This statement will print the settings we used to
    // the screen.
    cout << trainer << cropper << endl;

    // Now lets run the detector on the testing images and look at the outputs.  
    /*image_window win;
    for (auto&& img : images_test)
    {
        pyramid_up(img);
        auto dets = net(img);
        win.clear_overlay();
        win.set_image(img);
        for (auto&& d : dets)
            win.add_overlay(d);
        cin.get();
    }*/
    return 0;

    // Now that you finished this example, you should read dnn_mmod_train_find_cars_ex.cpp,
    // which is a more advanced example.  It discusses many issues surrounding properly
    // setting the MMOD parameters and creating a good training dataset.

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}




