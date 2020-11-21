#ifndef __IMAGE_DISPLAY_H__
#define __IMAGE_DISPLAY_H__


#include <vector>

#include "Utils/num_type.h"
#include "Utils/minimal_image.hpp"


namespace ds_slam
{
namespace visualizer
{
void DisplayImage(const char *windowName, const MinimalImageB *img, bool autoSize = false);
void DisplayImage(const char *windowName, const MinimalImageB3 *img, bool autoSize = false);
void DisplayImage(const char *windowName, const MinimalImageF *img, bool autoSize = false);
void DisplayImage(const char *windowName, const MinimalImageF3 *img, bool autoSize = false);
void DisplayImage(const char *windowName, const MinimalImageB16 *img, bool autoSize = false);

void DisplayImageStitch(const char *windowName, const std::vector<MinimalImageB *> images, int cc = 0, int rc = 0);
void DisplayImageStitch(const char *windowName, const std::vector<MinimalImageB3 *> images, int cc = 0, int rc = 0);
void DisplayImageStitch(const char *windowName, const std::vector<MinimalImageF *> images, int cc = 0, int rc = 0);
void DisplayImageStitch(const char *windowName, const std::vector<MinimalImageF3 *> images, int cc = 0, int rc = 0);

int WaitKey(int milliseconds);
void CloseAllWindows();
} // namespace visualizer
} // namespace ds_slam
#endif