#include "Visualizer/image_RW.h"

namespace ds_slam
{
namespace visualizer
{
MinimalImageB *ReadImageBW_8U(std::string filename)
{
    printf("not implemented. bye!\n");
    return 0;
};
MinimalImageB3 *ReadImageRGB_8U(std::string filename)
{
    printf("not implemented. bye!\n");
    return 0;
};
MinimalImage<unsigned short> *ReadImageBW_16U(std::string filename)
{
    printf("not implemented. bye!\n");
    return 0;
};
MinimalImageB *ReadStreamBW_8U(char *data, int numBytes)
{
    printf("not implemented. bye!\n");
    return 0;
};
void WriteImage(std::string filename, MinimalImageB *img){};
void WriteImage(std::string filename, MinimalImageB3 *img){};
void WriteImage(std::string filename, MinimalImageF *img){};
void WriteImage(std::string filename, MinimalImageF3 *img){};
} // namespace visualizer
} // namespace ds_slam
