#ifndef __DATASET_READER_H__
#define __DATASET_READER_H__


#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <boost/thread.hpp>

#include "Utils/undistort.hpp"
#include "Visualizer/image_RW.h"

#if HAS_ZIPLIB
#include "zip.h"
#endif

using namespace ds_slam;



inline int Getdir(std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        std::string name = std::string(dirp->d_name);

        if (name != "." && name != "..")
            files.push_back(name);
    }
    closedir(dp);

    std::sort(files.begin(), files.end());

    if (dir.at(dir.length() - 1) != '/')
        dir = dir + "/";
    for (unsigned int i = 0; i < files.size(); i++)
    {
        if (files[i].at(0) != '/')
            files[i] = dir + files[i];
    }

    return files.size();
}

struct PrepImageItem
{
    int id;
    bool isQueud;
    ImageAndExposure *pt;

    inline PrepImageItem(int _id)
    {
        id = _id;
        isQueud = false;
        pt = 0;
    }

    inline void release()
    {
        if (pt != 0)
            delete pt;
        pt = 0;
    }
};

class ImageFolderReader
{
public:
    /**
     * @brief Construct a new image folder reader object
     * @param images path 
     * @param calibration file 
     * @param gamma file 
     * @param vignette file 
     */
    ImageFolderReader(std::string path, std::string calibFile, std::string gammaFile, std::string vignetteFile)
    {
        this->path = path;
        this->calibfile = calibFile;

#if HAS_ZIPLIB
        ziparchive = 0;
        databuffer = 0;
#endif

        isZipped = (path.length() > 4 && path.substr(path.length() - 4) == ".zip");
        if (isZipped)
        {
#if HAS_ZIPLIB
            int ziperror = 0;
            ziparchive = zip_open(path.c_str(), ZIP_RDONLY, &ziperror);
            if (ziperror != 0)
            {
                printf("ERROR %d reading archive %s!\n", ziperror, path.c_str());
                exit(1);
            }

            files.clear();
            int numEntries = zip_get_num_entries(ziparchive, 0);
            for (int k = 0; k < numEntries; k++)
            {
                const char *name = zip_get_name(ziparchive, k, ZIP_FL_ENC_STRICT);
                std::string nstr = std::string(name);
                if (nstr == "." || nstr == "..")
                    continue;
                files.push_back(name);
            }

            printf("got %d entries and %d files!\n", numEntries, (int)files.size());
            std::sort(files.begin(), files.end());
#else
            printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
            exit(1);
#endif
        }
        else
            Getdir(path, files);

        // create undistorter
        undistort = Undistort::GetUndistorterForFile(calibFile, gammaFile, vignetteFile);
        widthOrg = undistort->GetOriginalSize()[0];
        heightOrg = undistort->GetOriginalSize()[1];
        width = undistort->GetSize()[0];
        height = undistort->GetSize()[1];

        // load timestamps if possible
        LoadTimestamps();
        printf("ImageFolderReader: got %d files in %s!\n", (int)files.size(), path.c_str());
    }

    /**
     * @brief Destroy the image folder Reader
     */
    ~ImageFolderReader()
    {
#if HAS_ZIPLIB
        if (ziparchive != 0)
            zip_close(ziparchive);
        if (databuffer != 0)
            delete databuffer;
#endif

        delete undistort;
    };

    /**
     * @brief get the original calibration
     */
    Eigen::VectorXf GetOriginalCalib()
    {
        return undistort->GetOriginalParameter().cast<float>();
    }

    /**
     * @brief get the original dimensions
     */
    Eigen::Vector2i GetOriginalDimensions()
    {
        return undistort->GetOriginalSize();
    }

    /**
     * @brief get the distort result
     * @param K intrinsic matrix
     * @param w width
     * @param h height
     */
    void GetCalibMono(Eigen::Matrix3f &K, int &w, int &h)
    {
        K = undistort->GetK().cast<float>();
        w = undistort->GetSize()[0];
        h = undistort->GetSize()[1];
    }

    /**
     * @brief set the global calibration
     */
    void SetGlobalCalibration()
    {
        int w_out, h_out;
        Eigen::Matrix3f K;
        GetCalibMono(K, w_out, h_out);
        SetGlobalCalib(w_out, h_out, K);
    }

    /**
     * @brief get the number images
     * @return int 
     */
    int GetNumImages()
    {
        return files.size();
    }

    /**
     * @brief get the timestamp of id
     * @param id id
     * @return double timestamp 
     */
    double GetTimestamp(int id)
    {
        if (timestamps.size() == 0)
            return id * 0.1f;
        if (id >= (int)timestamps.size())
            return 0;
        if (id < 0)
            return 0;
        return timestamps[id];
    }

    /**
     * @brief get the raw image with id
     * @param id 
     * @return MinimalImageB* raw image
     */
    MinimalImageB *GetImageRaw(int id)
    {
        return GetImageRaw_internal(id, 0);
    }

    /**
     * @brief get the image with id
     * @param id 
     * @param forceLoadDirectly (not used)
     * @return ImageAndExposure* image with exposure info
     */
    ImageAndExposure *GetImage(int id, bool forceLoadDirectly = false)
    {
        return GetImage_internal(id, 0);
    }

    /**
     * @brief get the photometric gamma
     * @return float* gamma map
     */
    inline float *GetPhotometricGamma()
    {
        if (undistort == 0 || undistort->photometricUndist == 0)
            return 0;
        return undistort->photometricUndist->GetG();
    }

    // undistorter. [0] always exists, [1-2] only when MT is enabled.
    Undistort *undistort;


private:
    // 
    MinimalImageB *GetImageRaw_internal(int id, int unused)
    {
        if (!isZipped)
        {
            // CHANGE FOR ZIP FILE
            return visualizer::ReadImageBW_8U(files[id]);
        }
        else
        {
#if HAS_ZIPLIB
            if (databuffer == 0)
                databuffer = new char[widthOrg * heightOrg * 6 + 10000];
            zip_file_t *fle = zip_fopen(ziparchive, files[id].c_str(), 0);
            long readbytes = zip_fread(fle, databuffer, (long)widthOrg * heightOrg * 6 + 10000);

            if (readbytes > (long)widthOrg * heightOrg * 6)
            {
                printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes, (long)widthOrg * heightOrg * 6 + 10000, files[id].c_str());
                delete[] databuffer;
                databuffer = new char[(long)widthOrg * heightOrg * 30];
                fle = zip_fopen(ziparchive, files[id].c_str(), 0);
                readbytes = zip_fread(fle, databuffer, (long)widthOrg * heightOrg * 30 + 10000);

                if (readbytes > (long)widthOrg * heightOrg * 30)
                {
                    printf("buffer still to small (read %ld/%ld). abort.\n", readbytes, (long)widthOrg * heightOrg * 30 + 10000);
                    exit(1);
                }
            }

            return visualizer::ReadStreamBW_8U(databuffer, readbytes);
#else
            printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
            exit(1);
#endif
        }
    }

    //
    ImageAndExposure *GetImage_internal(int id, int unused)
    {
        MinimalImageB *minimg = GetImageRaw_internal(id, 0);
        ImageAndExposure *ret2 = undistort->UndistortImage<unsigned char>(minimg,
                                                                          (exposures.size() == 0 ? 1.0f : exposures[id]),
                                                                          (timestamps.size() == 0 ? 0.0 : timestamps[id]));
        delete minimg;
        return ret2;
    }

    //
    inline void LoadTimestamps()
    {
        std::ifstream tr;
        std::string timesFile = path.substr(0, path.find_last_of('/')) + "/times.txt";
        tr.open(timesFile.c_str());
        while (!tr.eof() && tr.good())
        {
            std::string line;
            char buf[1000];
            tr.getline(buf, 1000);

            int id;
            double stamp;
            float exposure = 0;

            if (3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
            {
                timestamps.push_back(stamp);
                exposures.push_back(exposure);
            }

            else if (2 == sscanf(buf, "%d %lf", &id, &stamp))
            {
                timestamps.push_back(stamp);
                exposures.push_back(exposure);
            }
        }
        tr.close();

        // check if exposures are correct, (possibly skip)
        bool exposuresGood = ((int)exposures.size() == (int)GetNumImages());
        for (int i = 0; i < (int)exposures.size(); i++)
        {
            if (exposures[i] == 0)
            {
                // fix!
                float sum = 0, num = 0;
                if (i > 0 && exposures[i - 1] > 0)
                {
                    sum += exposures[i - 1];
                    num++;
                }
                if (i + 1 < (int)exposures.size() && exposures[i + 1] > 0)
                {
                    sum += exposures[i + 1];
                    num++;
                }

                if (num > 0)
                    exposures[i] = sum / num;
            }

            if (exposures[i] == 0)
                exposuresGood = false;
        }

        if ((int)GetNumImages() != (int)timestamps.size())
        {
            printf("set timestamps and exposures to zero!\n");
            exposures.clear();
            timestamps.clear();
        }

        if ((int)GetNumImages() != (int)exposures.size() || !exposuresGood)
        {
            printf("set EXPOSURES to zero!\n");
            exposures.clear();
        }

        printf("got %d images and %d timestamps and %d exposures.!\n", (int)GetNumImages(), (int)timestamps.size(), (int)exposures.size());
    }

    std::vector<ImageAndExposure *> preloadedImages;
    std::vector<std::string> files;
    std::vector<double> timestamps;
    std::vector<float> exposures;

    int width, height;
    int widthOrg, heightOrg;

    std::string path;
    std::string calibfile;

    bool isZipped;

#if HAS_ZIPLIB
    zip_t *ziparchive;
    char *databuffer;
#endif

};

#endif