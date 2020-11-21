#include "Visualizer/image_display.h"

namespace ds_slam
{
namespace visualizer
{
void DisplayImage(const char* windowName, const MinimalImageB* img, bool autoSize) {};
void DisplayImage(const char* windowName, const MinimalImageB3* img, bool autoSize) {};
void DisplayImage(const char* windowName, const MinimalImageF* img, bool autoSize) {};
void DisplayImage(const char* windowName, const MinimalImageF3* img, bool autoSize) {};
void DisplayImage(const char* windowName, const MinimalImageB16* img, bool autoSize) {};

void DisplayImageStitch(const char* windowName, const std::vector<MinimalImageB*> images, int cc, int rc) {};
void DisplayImageStitch(const char* windowName, const std::vector<MinimalImageB3*> images, int cc, int rc) {};
void DisplayImageStitch(const char* windowName, const std::vector<MinimalImageF*> images, int cc, int rc) {};
void DisplayImageStitch(const char* windowName, const std::vector<MinimalImageF3*> images, int cc, int rc) {};

int WaitKey(int milliseconds) {return 0;};
void CloseAllWindows() {};
} // namespace visualizer
} // namespace ds_slam