#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <locale.h>
#include <signal.h>
#include <unistd.h>
#include <boost/thread.hpp>


#include "Utils/settings.h"
#include "Utils/num_type.h"
#include "Utils/global_calib.h"
#include "Utils/dataset_reader.h"
#include "FullSystem/full_system.hpp"


#include "Visualizer/visualizer_3D.hpp"
#include "Visualizer/image_display.h"
#include "Visualizer/Pangolin/visualizer_pangolin.hpp"
#include "Visualizer/visualizer_sample.hpp"


// parameters
std::string source = "";
std::string calib = "";
std::string vignette = "";
std::string gammaCalib = "";
std::string vocFile = "";
int mode = 0;
bool reversePlay = false;
bool preload = false;
int startID = 0;
int endID = 100000;
float playbackSpeed = 0;
bool useSampleOutput = false;

using namespace ds_slam;


void SettingsDefault(int preset)
{
    printf("\n=============== PRESET Settings: ===============\n");
    if (preset == 0 || preset == 1)
    {
        printf("DEFAULT settings:\n"
               "- %s real-time enforcing\n"
               "- 2000 active points\n"
               "- 5-7 active frames\n"
               "- 1-6 LM iteration each KF\n"
               "- original image resolution\n",
               preset == 0 ? "no " : "1x");

        playbackSpeed = (preset == 0 ? 0 : 1);
        preload = preset == 1;
        setting_desiredImmatureDensity = 1500;
        setting_desiredPointDensity = 2000;
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_maxOptIterations = 6;
        setting_minOptIterations = 1;

        setting_logStuff = false;
    }

    if (preset == 2 || preset == 3)
    {
        printf("FAST settings:\n"
               "- %s real-time enforcing\n"
               "- 800 active points\n"
               "- 4-6 active frames\n"
               "- 1-4 LM iteration each KF\n"
               "- 424 x 320 image resolution\n",
               preset == 0 ? "no " : "5x");

        playbackSpeed = (preset == 2 ? 0 : 5);
        preload = preset == 3;
        setting_desiredImmatureDensity = 600;
        setting_desiredPointDensity = 800;
        setting_minFrames = 4;
        setting_maxFrames = 6;
        setting_maxOptIterations = 4;
        setting_minOptIterations = 1;

        benchmarkSetting_width = 424;
        benchmarkSetting_height = 320;

        setting_logStuff = false;
    }

    printf("==============================================\n");
}

void ParseArgument(char *arg)
{
    int option;
    float foption;
    char buf[1000];

    // input configuration
    if (1 == sscanf(arg, "files=%s", buf))
    {
        source = buf;
        printf("loading data from %s!\n", source.c_str());
        return;
    }
    if (1 == sscanf(arg, "calib=%s", buf))
    {
        calib = buf;
        printf("loading calibration from %s!\n", calib.c_str());
        return;
    }
    if (1 == sscanf(arg, "vignette=%s", buf))
    {
        vignette = buf;
        printf("loading vignette from %s!\n", vignette.c_str());
        return;
    }
    if (1 == sscanf(arg, "gamma=%s", buf))
    {
        gammaCalib = buf;
        printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
        return;
    }
    if (1 == sscanf(arg, "preset=%d", &option))
    {
        SettingsDefault(option);
        return;
    }
    if (1 == sscanf(arg, "mode=%d", &option))
    {
        mode = option;
        if (option == 0)
        {
            printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
        }
        if (option == 1)
        {
            printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
        }
        if (option == 2)
        {
            printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
            setting_photometricCalibration = 0;
            setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd = 3;
        }
        return;
    }
    if (1 == sscanf(arg, "dbow=%s", buf))
    {
        vocFile = buf;
        printf("loading dbow from %s!\n", vocFile.c_str());
        return;
    }

    // input data configuration
    if (1 == sscanf(arg, "reverse=%d", &option))
    {
        if (option == 1)
        {
            reversePlay = true;
            printf("reverse play data!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "preload=%d", &option))
    {
        if (option == 1)
        {
            preload = true;
            printf("preload data!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "start=%d", &option))
    {
        startID = option;
        printf("start at %d!\n", startID);
        return;
    }
    if (1 == sscanf(arg, "end=%d", &option))
    {
        endID = option;
        printf("end at %d!\n", endID);
        return;
    }
    if (1 == sscanf(arg, "speed=%f", &foption))
    {
        playbackSpeed = foption;
        printf("playback speed %f!\n", playbackSpeed);
        return;
    }

    // print configuration
    if (1 == sscanf(arg, "quiet=%d", &option))
    {
        if (option == 1)
        {
            setting_debugout_runquiet = true;
            printf("quiet mode, I'll shut up!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "nolog=%d", &option))
    {
        if (option == 1)
        {
            setting_logStuff = false;
            printf("disable logging!\n");
        }
        return;
    }

    // display configuration
    if (1 == sscanf(arg, "nogui=%d", &option))
    {
        if (option == 1)
        {
            disableAllDisplay = true;
            printf("no GUI!\n");
        }
        return;
    }
    if (1 == sscanf(arg, "sampleoutput=%d", &option))
    {
        if (option == 1)
        {
            useSampleOutput = true;
            printf("using sample output!\n");
        }
        return;
    }

    // multi-thread
    if (1 == sscanf(arg, "nomt=%d", &option))
    {
        if (option == 1)
        {
            multiThreading = false;
            printf("NO MultiThreading!\n");
        }
        return;
    }

    // save image for debug
    if (1 == sscanf(arg, "save=%d", &option))
    {
        if (option == 1)
        {
            debugSaveImages = true;
            if (42 == system("rm -rf images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if (42 == system("mkdir images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if (42 == system("rm -rf images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            if (42 == system("mkdir images_out"))
                printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
            printf("SAVE IMAGES!\n");
        }
        return;
    }

    printf("could not parse argument \"%s\"!!!!\n", arg);
}


int main(int argc, char **argv)
{
    // parse arguments
    for (int i = 1; i < argc; i++)
        ParseArgument(argv[i]);

    // check parameters
    if(source == "" || calib == "")
    {
        printf("ERROR: must be configure image and calibration file!\n");
        return -1;
    }

    // create image reader and set system global calibration result
    ImageFolderReader *reader = new ImageFolderReader(source, calib, gammaCalib, vignette);
    reader->SetGlobalCalibration();
    if (setting_photometricCalibration > 0 && reader->GetPhotometricGamma() == 0)
    {
        printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2\n");
        return -1;
    }

    int lstart = startID;
    int lend = endID;
    int linc = 1;
    if (reversePlay)
    {
        printf("reverse play!!!!");
        lstart = endID - 1;
        if (lstart >= reader->GetNumImages())
            lstart = reader->GetNumImages() - 1;
        lend = startID;
        linc = -1;
    }

    // create slam system
    FullSystem *fullSystem = new FullSystem(vocFile);
    fullSystem->SetGammaFunction(reader->GetPhotometricGamma());
    fullSystem->linearizeOperation = (playbackSpeed == 0);

    visualizer::ViewerPangolin *viewer = 0;
    if (!disableAllDisplay)
    {
        viewer = new visualizer::ViewerPangolin(wG[0], hG[0], false);
        fullSystem->viewers.push_back(viewer);
    }

    if (useSampleOutput)
        fullSystem->viewers.push_back(new visualizer::ViewerSample());

    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]() {
        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;
        for (int i = lstart; i >= 0 && i < reader->GetNumImages() && linc * i < linc * lend; i += linc)
        {
            idsToPlay.push_back(i);
            if (timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back((double)0);
            }
            else
            {
                double tsThis = reader->GetTimestamp(idsToPlay[idsToPlay.size() - 1]);
                double tsPrev = reader->GetTimestamp(idsToPlay[idsToPlay.size() - 2]);
                timesToPlayAt.push_back(timesToPlayAt.back() + fabs(tsThis - tsPrev) / playbackSpeed);
            }
        }

        std::vector<ImageAndExposure *> preloadedImages;
        if (preload)
        {
            printf("LOADING ALL IMAGES!\n");
            for (int ii = 0; ii < (int)idsToPlay.size(); ii++)
            {
                int i = idsToPlay[ii];
                preloadedImages.push_back(reader->GetImage(i));
            }
        }

        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset = 0;

        for (int ii = 0; ii < (int)idsToPlay.size(); ii++)
        {
            if (!fullSystem->initialized) // if not initialized: reset start time.
            {
                gettimeofday(&tv_start, NULL);
                started = clock();
                sInitializerOffset = timesToPlayAt[ii];
            }

            int i = idsToPlay[ii];

            ImageAndExposure *img;
            if (preload)
                img = preloadedImages[ii];
            else
                img = reader->GetImage(i);

            bool skipFrame = false;
            if (playbackSpeed != 0)
            {
                struct timeval tv_now;
                gettimeofday(&tv_now, NULL);
                double sSinceStart = sInitializerOffset + ((tv_now.tv_sec - tv_start.tv_sec) + (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));

                if (sSinceStart < timesToPlayAt[ii])
                    usleep((int)((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000));
                else if (sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2))
                {
                    printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                    skipFrame = true;
                }
            }

            if (!skipFrame)
                fullSystem->AddActiveFrame(img, i);

            delete img;

            if (fullSystem->initFailed || setting_fullResetRequested)
            {
                if (ii < 250 || setting_fullResetRequested)
                {
                    printf("RESETTING!\n");

                    std::vector<visualizer::Visualizer3D *> wraps = fullSystem->viewers;
                    delete fullSystem;

                    for (visualizer::Visualizer3D *ow : wraps)
                        ow->Reset();

                    fullSystem = new FullSystem(vocFile);
                    fullSystem->SetGammaFunction(reader->GetPhotometricGamma());
                    fullSystem->linearizeOperation = (playbackSpeed == 0);

                    fullSystem->viewers = wraps;

                    setting_fullResetRequested = false;
                }
            }

            if (fullSystem->isLost)
            {
                printf("LOST!!\n");
                break;
            }
        }
        fullSystem->BlockUntilMappingIsFinished();
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);

        fullSystem->PrintResult("result.txt");

        int numFramesProcessed = abs(idsToPlay[0] - idsToPlay.back());
        double numSecondsProcessed = fabs(reader->GetTimestamp(idsToPlay[0]) - reader->GetTimestamp(idsToPlay.back()));
        double MilliSecondsTakenSingle = 1000.0f * (ended - started) / (float)(CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
        printf("\n======================"
               "\n%d Frames (%.1f fps)"
               "\n%.2fms per frame (single core); "
               "\n%.2fms per frame (multi core); "
               "\n%.3fx (single core); "
               "\n%.3fx (multi core); "
               "\n======================\n\n",
               numFramesProcessed, numFramesProcessed / numSecondsProcessed,
               MilliSecondsTakenSingle / numFramesProcessed,
               MilliSecondsTakenMT / (float)numFramesProcessed,
               1000 / (MilliSecondsTakenSingle / numSecondsProcessed),
               1000 / (MilliSecondsTakenMT / numSecondsProcessed));
        //fullSystem->PrintFrameLifetimes();
        if (setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f * (ended - started) / (float)(CLOCKS_PER_SEC * reader->GetNumImages()) << " "
                  << ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) / (float)reader->GetNumImages() << "\n";
            tmlog.flush();
            tmlog.close();
        }
    });

    if (viewer != 0)
        viewer->Run();

    runthread.join();

    for (visualizer::Visualizer3D *ow : fullSystem->viewers)
    {
        ow->Join();
        delete ow;
    }

    printf("DELETE FULLSYSTEM!\n");
    delete fullSystem;

    printf("DELETE READER!\n");
    delete reader;

    printf("EXIT NOW!\n");
    return 0;
}
