#ifndef __IMAGE_RW_H__
#define __IMAGE_RW_H__

#include "Utils/num_type.h"
#include "Utils/minimal_image.hpp"

namespace ds_slam
{
namespace visualizer
{
MinimalImageB *ReadImageBW_8U(std::string filename);
MinimalImageB3 *ReadImageRGB_8U(std::string filename);
MinimalImage<unsigned short> *ReadImageBW_16U(std::string filename);
MinimalImageB *ReadStreamBW_8U(char *data, int numBytes);

void WriteImage(std::string filename, MinimalImageB *img);
void WriteImage(std::string filename, MinimalImageB3 *img);
void WriteImage(std::string filename, MinimalImageF *img);
void WriteImage(std::string filename, MinimalImageF3 *img);
} // namespace visualizer
} // namespace ds_slam

#endif