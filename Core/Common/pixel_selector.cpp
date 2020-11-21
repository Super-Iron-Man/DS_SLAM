#include "Common/pixel_selector.hpp"
#include "Visualizer/image_display.h"
#include "Utils/misc.h"


namespace ds_slam
{

PixelSelector::PixelSelector(int w, int h)
{
    randomPattern = new unsigned char[w * h];
    std::srand(3141592); // want to be deterministic.
    for (int i = 0; i < w * h; i++)
        randomPattern[i] = rand() & 0xFF; // random pattern, so DSO is not constant output at all

    currentPotential = 3;

    gradHist = new int[100 * (1 + w / 32) * (1 + h / 32)];
    ths = new float[(w / 32) * (h / 32) + 100];
    thsSmoothed = new float[(w / 32) * (h / 32) + 100];

    allowFast = false;
    gradHistFrame = 0;
}

PixelSelector::~PixelSelector()
{
    delete[] randomPattern;
    delete[] gradHist;
    delete[] ths;
    delete[] thsSmoothed;
}

int computeHistQuantil(int *hist, float below)
{
    int th = hist[0] * below + 0.5f;
    for (int i = 0; i < 90; i++)
    {
        th -= hist[i + 1];
        if (th < 0)
            return i;
    }
    return 90;
}

void PixelSelector::MakeHists(const FrameHessian *const fh)
{
    gradHistFrame = fh;
    float *mapmax0 = fh->absSquaredGrad[0];

    int w = wG[0];
    int h = hG[0];

    int w32 = w / 32; // block size 32*32
    int h32 = h / 32;
    thsStep = w32;

    for (int y = 0; y < h32; y++)
    {
        for (int x = 0; x < w32; x++)
        {
            float *map0 = mapmax0 + 32 * x + 32 * y * w;
            int *hist0 = gradHist;
            memset(hist0, 0, sizeof(int) * 50); // 50 bins

            for (int j = 0; j < 32; j++)
                for (int i = 0; i < 32; i++)
                {
                    int it = i + 32 * x;
                    int jt = j + 32 * y;
                    if (it > w - 2 || jt > h - 2 || it < 1 || jt < 1)
                        continue;
                    int g = sqrtf(map0[i + j * w]);
                    if (g > 48)
                        g = 48;
                    hist0[g + 1]++;
                    hist0[0]++;
                }

            ths[x + y * w32] = computeHistQuantil(hist0, setting_minGradHistCut) + setting_minGradHistAdd; // caculate the thresh of block
        }
    }

    // smmoth 3*3 block
    for (int y = 0; y < h32; y++)
    {
        for (int x = 0; x < w32; x++)
        {
            float sum = 0, num = 0;
            if (x > 0)
            {
                if (y > 0)
                {
                    num++;
                    sum += ths[x - 1 + (y - 1) * w32]; // block (x-1, y-1)
                }
                if (y < h32 - 1)
                {
                    num++;
                    sum += ths[x - 1 + (y + 1) * w32]; // block (x-1, y+1)
                }
                num++;
                sum += ths[x - 1 + (y)*w32]; // block (x-1, y)
            }

            if (x < w32 - 1)
            {
                if (y > 0)
                {
                    num++;
                    sum += ths[x + 1 + (y - 1) * w32]; // block (x+1, y-1)
                }
                if (y < h32 - 1)
                {
                    num++;
                    sum += ths[x + 1 + (y + 1) * w32]; // block (x+1, y+1)
                }
                num++;
                sum += ths[x + 1 + (y)*w32]; // block (x+1, y)
            }

            if (y > 0)
            {
                num++;
                sum += ths[x + (y - 1) * w32]; // block (x, y-1)
            }
            if (y < h32 - 1)
            {
                num++;
                sum += ths[x + (y + 1) * w32]; // block (x, y+1)
            }
            num++;
            sum += ths[x + y * w32]; // block (x, y)

            thsSmoothed[x + y * w32] = (sum / num) * (sum / num); // average, create smoothed thresh map of blocks
        }
    }
}

int PixelSelector::MakeMaps(const FrameHessian *const fh, int *map_out, int *corner_out,
                            float density, int recursionsLeft, bool plot, float thFactor)
{
    float numHave = 0;
    float numWant = density;
    float quotia;
    int idealPotential = currentPotential;

    //if(setting_pixelSelectionUseFast>0 && allowFast)
    //{
    //    memset(map_out, 0, sizeof(float)*wG[0]*hG[0]);
    //    std::vector<cv::KeyPoint> pts;
    //    cv::Mat img8u(hG[0],wG[0],CV_8U);
    //    for(int i=0;i<wG[0]*hG[0];i++)
    //    {
    //        float v = fh->dI[i][0]*0.8;
    //        img8u.at<uchar>(i) = (!std::isfinite(v) || v>255) ? 255 : v;
    //    }
    //    cv::FAST(img8u, pts, setting_pixelSelectionUseFast, true);
    //    for(unsigned int i=0;i<pts.size();i++)
    //    {
    //        int x = pts[i].pt.x+0.5;
    //        int y = pts[i].pt.y+0.5;
    //        map_out[x+y*wG[0]]=1;
    //        numHave++;
    //    }
    //
    //    printf("FAST selection: got %f / %f!\n", numHave, numWant);
    //    quotia = numWant / numHave;
    //}
    //else
    {

        // the number of selected pixels behaves approximately as
        // K / (pot+1)^2, where K is a scene-dependent constant.
        // we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.

        if (fh != gradHistFrame)
            MakeHists(fh);

        // select!
        Eigen::Vector3i n = this->Select(fh, map_out, corner_out, currentPotential, thFactor);

        // sub-select!
        numHave = n[0] + n[1] + n[2];
        quotia = numWant / numHave;

        // by default we want to over-sample by 40% just to be sure.
        float K = numHave * (currentPotential + 1) * (currentPotential + 1);
        idealPotential = sqrtf(K / numWant) - 1; // round down.
        if (idealPotential < 1)
            idealPotential = 1;

        if (recursionsLeft > 0 && quotia > 1.25 && currentPotential > 1)
        {
            // re-sample to get more points!
            // potential needs to be smaller
            if (idealPotential >= currentPotential)
                idealPotential = currentPotential - 1;

            //printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
            //        100*numHave/(float)(wG[0]*hG[0]),
            //        100*numWant/(float)(wG[0]*hG[0]),
            //        currentPotential,
            //        idealPotential);
            currentPotential = idealPotential;
            return MakeMaps(fh, map_out, corner_out, density, recursionsLeft - 1, plot, thFactor);
        }
        else if (recursionsLeft > 0 && quotia < 0.25)
        {
            // re-sample to get less points!
            // potential needs to be bigger
            if (idealPotential <= currentPotential)
                idealPotential = currentPotential + 1;

            //printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
            //        100*numHave/(float)(wG[0]*hG[0]),
            //        100*numWant/(float)(wG[0]*hG[0]),
            //        currentPotential,
            //        idealPotential);
            currentPotential = idealPotential;
            return MakeMaps(fh, map_out, corner_out, density, recursionsLeft - 1, plot, thFactor);
        }
    }

    int numHaveSub = numHave;
    if (quotia < 0.95)
    {
        int wh = wG[0] * hG[0];
        int rn = 0;
        unsigned char charTH = 255 * quotia;
        for (int i = 0; i < wh; i++)
        {
            if (map_out[i] != 0)
            {
                if (randomPattern[rn] > charTH)
                {
                    map_out[i] = 0;
                    numHaveSub--;
                }
                rn++;
            }
        }
    }

    //printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
    //        100*numHave/(float)(wG[0]*hG[0]),
    //        100*numWant/(float)(wG[0]*hG[0]),
    //        currentPotential,
    //        idealPotential,
    //        100*numHaveSub/(float)(wG[0]*hG[0]));
    currentPotential = idealPotential;

    if (plot)
    {
        int w = wG[0];
        int h = hG[0];

        MinimalImageB3 img(w, h);

        for (int i = 0; i < w * h; i++)
        {
            float c = fh->dI[i][0] * 0.7;
            if (c > 255)
                c = 255;
            img.at(i) = Vec3b(c, c, c);
        }
        visualizer::DisplayImage("Selector Image", &img);

        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int i = x + y * w;
                if (map_out[i] == 1)
                    img.SetPixelCirc(x, y, Vec3b(0, 255, 0));
                else if (map_out[i] == 2)
                    img.SetPixelCirc(x, y, Vec3b(255, 0, 0));
                else if (map_out[i] == 4)
                    img.SetPixelCirc(x, y, Vec3b(0, 0, 255));

                if (corner_out[i] == 1)
                    img.SetPixelCirc(x, y, Vec3b(255, 0, 255));
            }
        visualizer::DisplayImage("Selector Pixels", &img);

        while (true)
        {
            char k = visualizer::WaitKey(0);
            if (k == ' ')
                break;
            handleKey(k);
        }
    }

    return numHaveSub;
}

Eigen::Vector3i PixelSelector::Select(const FrameHessian *const fh, int *map_out, int *corner_out,
                                      int pot, float thFactor)
{
    Eigen::Vector3f const *const map0 = fh->dI; // select level = 0 image pyramid

    float *mapmax0 = fh->absSquaredGrad[0]; // the square gradent pyramid map
    float *mapmax1 = fh->absSquaredGrad[1];
    float *mapmax2 = fh->absSquaredGrad[2];

    int w = wG[0];
    int w1 = wG[1];
    int w2 = wG[2];
    int h = hG[0];

    // 16 directions
    const Vec2f directions[16] = {
        Vec2f(0, 1.0000),
        Vec2f(0.3827, 0.9239),
        Vec2f(0.1951, 0.9808),
        Vec2f(0.9239, 0.3827),
        Vec2f(0.7071, 0.7071),
        Vec2f(0.3827, -0.9239),
        Vec2f(0.8315, 0.5556),
        Vec2f(0.8315, -0.5556),
        Vec2f(0.5556, -0.8315),
        Vec2f(0.9808, 0.1951),
        Vec2f(0.9239, -0.3827),
        Vec2f(0.7071, -0.7071),
        Vec2f(0.5556, 0.8315),
        Vec2f(0.9808, -0.1951),
        Vec2f(1.0000, 0.0000),
        Vec2f(0.1951, -0.9808)};

    memset(map_out, 0, w * h * sizeof(/*PixelSelectorStatus*/int)); // reset the map
    memset(corner_out, 0, w * h * sizeof(int));

    float dw1 = setting_gradDownweightPerLevel;
    float dw2 = dw1 * dw1;

    // create big block(4*4 sub-block) as pyramid, caculate in each sub-block，
    int n2 = 0, n3 = 0, n4 = 0;
    float maxScore = 0;
    float *scoreMap = new float[w * h];
    memset(scoreMap, 0, w * h * sizeof(float));
    for (int y4 = 0; y4 < h; y4 += (4 * pot))
    {
        for (int x4 = 0; x4 < w; x4 += (4 * pot))
        {
            int my3 = std::min((4 * pot), h - y4);
            int mx3 = std::min((4 * pot), w - x4);
            int bestIdx4 = -1;
            float bestVal4 = 0;
            Vec2f dir4 = directions[randomPattern[n2] & 0xF]; // direction of level 2
            for (int y3 = 0; y3 < my3; y3 += (2 * pot))
            {
                for (int x3 = 0; x3 < mx3; x3 += (2 * pot))
                {
                    int x34 = x3 + x4;
                    int y34 = y3 + y4;
                    int my2 = std::min((2 * pot), h - y34);
                    int mx2 = std::min((2 * pot), w - x34);
                    int bestIdx3 = -1;
                    float bestVal3 = 0;
                    Vec2f dir3 = directions[randomPattern[n2] & 0xF]; // direction of level 1
                    for (int y2 = 0; y2 < my2; y2 += pot)
                    {
                        for (int x2 = 0; x2 < mx2; x2 += pot)
                        {
                            int x234 = x2 + x34;
                            int y234 = y2 + y34;
                            int my1 = std::min(pot, h - y234);
                            int mx1 = std::min(pot, w - x234);
                            int bestIdx2 = -1;
                            float bestVal2 = 0;
                            Vec2f dir2 = directions[randomPattern[n2] & 0xF]; // direction of level 0

                            // caculate in a sub-block
                            for (int y1 = 0; y1 < my1; y1 += 1)
                            {
                                for (int x1 = 0; x1 < mx1; x1 += 1)
                                {
                                    assert(x1 + x234 < w);
                                    assert(y1 + y234 < h);
                                    int idx = x1 + x234 + w * (y1 + y234); // 
                                    int xf = x1 + x234;
                                    int yf = y1 + y234;

                                    if (xf < 4 || xf >= w - 5 || yf < 4 || yf > h - 4)
                                        continue;

                                    float pixelTH0 = thsSmoothed[(xf >> 5) + (yf >> 5) * thsStep]; // selecte the thresh of block
                                    float pixelTH1 = pixelTH0 * dw1;
                                    float pixelTH2 = pixelTH1 * dw2;

                                    // level 0
                                    float ag0 = mapmax0[idx]; // the square gradent
                                    if (ag0 > pixelTH0 * thFactor) // condition: gradent is larger than thresh
                                    {
                                        Vec2f ag0d = map0[idx].tail<2>(); // [dx, dy]
                                        float dirNorm = fabsf((float)(ag0d.dot(dir2))); // add uniform direction distribution with 
                                        if (!setting_selectDirectionDistribution) // if no direction distribution, only consider gradent magnitude
                                            dirNorm = ag0;

                                        if (dirNorm > bestVal2) // select best
                                        {
                                            bestVal2 = dirNorm;
                                            bestIdx2 = idx;
                                            bestIdx3 = -2;
                                            bestIdx4 = -2;
                                        }
                                    }
                                    if (bestIdx3 == -2)
                                        continue;

                                    // level 1
                                    float ag1 = mapmax1[(int)(xf * 0.5f + 0.25f) + (int)(yf * 0.5f + 0.25f) * w1];
                                    if (ag1 > pixelTH1 * thFactor)
                                    {
                                        Vec2f ag0d = map0[idx].tail<2>(); // [dx, dy] at level 0
                                        float dirNorm = fabsf((float)(ag0d.dot(dir3)));
                                        if (!setting_selectDirectionDistribution)
                                            dirNorm = ag1;

                                        if (dirNorm > bestVal3)
                                        {
                                            bestVal3 = dirNorm;
                                            bestIdx3 = idx;
                                            bestIdx4 = -2;
                                        }
                                    }
                                    if (bestIdx4 == -2)
                                        continue;

                                    // level 2
                                    float ag2 = mapmax2[(int)(xf * 0.25f + 0.125) + (int)(yf * 0.25f + 0.125) * w2];
                                    if (ag2 > pixelTH2 * thFactor)
                                    {
                                        Vec2f ag0d = map0[idx].tail<2>(); // [dx, dy] at level 0
                                        float dirNorm = fabsf((float)(ag0d.dot(dir4)));
                                        if (!setting_selectDirectionDistribution)
                                            dirNorm = ag2;

                                        if (dirNorm > bestVal4)
                                        {
                                            bestVal4 = dirNorm;
                                            bestIdx4 = idx;
                                        }
                                    }
                                }
                            }

                            // select level 0
                            if (bestIdx2 > 0)
                            {
                                map_out[bestIdx2] = 1; // flag level
                                bestVal3 = 1e10;
                                n2++;

                                // caculate the corner score
                                int realX = bestIdx2 % w;
                                int realY = bestIdx2 / w;
                                float s = ShiTomasiScore(fh, realX, realY);
                                scoreMap[bestIdx2] = s;
                                if (s > maxScore)
                                    maxScore = s;
                            }
                        }
                    }

                    // select level 1
                    if (bestIdx3 > 0)
                    {
                        map_out[bestIdx3] = 2; // flag level
                        bestVal4 = 1e10;
                        n3++;
                    }
                }
            }

            // select level 2
            if (bestIdx4 > 0)
            {
                map_out[bestIdx4] = 4; // flag level
                n4++;
            }
        }
    }

    // judge the corner
    for(int id = 0; id < w * h; id++)
    {
        if(scoreMap[id] > 0.01 * maxScore)
            corner_out[id] = 1;
    }
    delete[] scoreMap;

    return Eigen::Vector3i(n2, n3, n4);
}

float PixelSelector::ShiTomasiScore(const FrameHessian *fh, int u, int v, int halfbox, int level)
{

    float dXX = 0.0;
    float dYY = 0.0;
    float dXY = 0.0;

    const int box_size = 2 * halfbox;
    const int box_area = box_size * box_size;
    const int x_min = u - halfbox;
    const int x_max = u + halfbox;
    const int y_min = v - halfbox;
    const int y_max = v + halfbox;

    if (x_min < 1 || x_max >= wG[level] - 1 || y_min < 1 || y_max >= hG[level] - 1)
        return 0.0; // patch is too close to the boundary
    const int stride = wG[level];

    for (int y = y_min; y < y_max; ++y)
    {
        for (int x = x_min; x < x_max; ++x)
        {
            float dx = fh->dIp[level][y * stride + x][1];
            float dy = fh->dIp[level][y * stride + x][2];
            dXX += dx * dx;
            dYY += dy * dy;
            dXY += dx * dy;
        }
    }

    // Find and return smaller eigenvalue:
    dXX = dXX / (2.0 * box_area);
    dYY = dYY / (2.0 * box_area);
    dXY = dXY / (2.0 * box_area);
    return 0.5 * (dXX + dYY - sqrt((dXX + dYY) * (dXX + dYY) - 4 * (dXX * dYY - dXY * dXY)));
}


}